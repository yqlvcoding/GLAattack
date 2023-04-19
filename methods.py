import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
import os
import random
import kornia as K
from models import get_source_model
from utils import patch_level_aug_here


class Base_all(object):
    '''
    Base class of all
    '''
    def __init__(self, args):
        self.device = args.device
        self.epsilon = args.epsilon
        self.steps = args.steps
        self.step_size = self.epsilon / self.steps
        self.init_adam_lr = args.init_adam_lr
        
        self.activion_layer = args.mid_layer
        self.features_hook = dict()
        # change multi_step_size 
        if args.multi_step_size:
            self.step_size *= args.multi_step_size

        # use_adam: using adam to update noise
        self.use_adam = args.use_adam
        print("use_adam", self.use_adam)
        
        
        # loading model
        self.model_name = args.generate_model_name
        self.l2reg = args.l2reg
        self.l2reg_alpha = args.l2reg_alpha
        if args.attack != 'RandomMaxAttack':
            self.model = get_source_model(args)

        if args.mean is not None and args.std is not None:
            self.mean = torch.tensor(args.mean).to(self.device)
            self.std = torch.tensor(args.std).to(self.device)
        else:
            self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)
            self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)

        # get mid_layer output
        if args.attack != 'RandomMaxAttack' and self.activion_layer is not None or self.model_name == 'RN50':
            self.handle = self._make_hook()
        

    def forward(self, *input):
        raise NotImplementedError

    def _make_hook(self):
        handle = None
        if self.model_name == "ViT-B/16" or self.model_name == 'ViT-B/32':
            handle = self.model.visual.transformer.resblocks[self.activion_layer].register_forward_hook(self._getActivation("layer{0}".format(self.activion_layer)))
        elif self.model_name == "RN50":
            handle = self.model.visual.layer3.register_forward_hook(self._getActivation('layer3'))
        return handle

    def _getActivation(self, name):
        def hook(model, input, output):
            self.features_hook[name] = output
        return hook

    def _remove_handle(self, handle):
        handle.remove()

    def _save_images(self, unnormed_inps, filenames, output_dir):
        for i,filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            save_image(unnormed_inps[i], save_path)
        
    def _unnorm(self, x):
        tmpx = torch.clone(x)
        tmpx.mul_(self.std[:, None, None]).add_(self.mean[:, None, None])
        return tmpx

    def _norm(self, x):
        tmpx = torch.clone(x)
        tmpx.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return tmpx

    def _get_middle_feature(self):
        if self.model_name == "ViT-B/16" or self.model_name == 'ViT-B/32':
            mid_feature = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)
        elif self.model_name == "RN50":
            mid_feature = self.features_hook["layer3"]
        return mid_feature
    
    
    def _make_perts(self, shape, rand_init=0):
        if rand_init:
            tn = torch.rand(*shape).to(self.device)
            tn = (tn - 0.5) * 2 * self.epsilon     # [-self.epsilon, self.epsilon]
            perts = tn
        else:
            perts = torch.zeros(*shape).to(self.device)
        perts.requires_grad_()
        if self.use_adam:
            perts = torch.nn.Parameter(perts, requires_grad=True)
            self.optimizer = torch.optim.Adam([perts], lr=self.init_adam_lr)
            self.optimizer.zero_grad()
        return perts
    
    def _go(self, loss, perts):
        loss.backward()
        if self.use_adam:
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:    
            grad = perts.grad.data
            perts.data = self._update_perts(perts.data, grad)
            perts.grad.data.zero_()

    def _get_adv(self, unnorm, perts):
        return torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1).detach()

    def _update_perts(self, perts, grad):
        perts = perts - self.step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts
    
    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images


class DRstd(Base_all):
    '''
    std method: to reduce middle layer's feature std.
    '''
    def __init__(self, args):
        super(DRstd, self).__init__(args)
    
    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        perts = self._make_perts((bs, C, H, W), rand_init=1)

        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            self.model.encode_image(nowx)
            features = self._get_middle_feature()
            features = features.view(bs, -1)
            #features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2).view(bs, -1)   # LND -> NLD
            loss = torch.sum(features.std(dim=-1))
            
            self._go(loss, perts)

        return self._get_adv(unnorm, perts)


class I2V(Base_all):
    '''
    Using middle layer's features to calculate cos loss, without label information.
    '''
    def __init__(self, args):
        super(I2V, self).__init__(args)
        
    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        self.model.encode_image(normed_inps)         # forward for get ori_activation
        #ori_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
        ori_features = self._get_middle_feature()
        ori_features = ori_features.view(bs, -1)
        perts = self._make_perts((bs, C, H, W), rand_init=1)
        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = F.cosine_similarity(now_features.float(), ori_features.float()) 
            loss = torch.sum(this_loss)
            self._go(loss, perts)
        return self._get_adv(unnorm, perts)


class DIM_cos(Base_all):
    def __init__(self, args):
        super(DIM_cos, self).__init__(args)
    
    def _input_diversity(self, images):
        if random.random() < 0.5:
            return images
        else:
            rnd = torch.randint(224,250, size=(1,1)).item()
            rescaled = images.view((-1, ) + images.shape[1:])
            rescaled = torch.nn.functional.interpolate(rescaled, size=[rnd, rnd], mode='nearest')
            h_rem = 250 - rnd
            w_rem = 250 - rnd
            pad_top = torch.randint(0, h_rem, size=(1,1)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_rem, size=(1,1)).item()
            pad_right = w_rem - pad_left
            padded = nn.functional.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom])
            # return torchvision.transforms.functional.resize(padded,[224, 224], Image.NEAREST)
            padded = torch.nn.functional.interpolate(padded, size=[224, 224], mode='nearest')
            padded = padded.view(images.shape)
            return padded

    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        self.model.encode_image(normed_inps)
        ori_features = self._get_middle_feature()
        #ori_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
        ori_features = ori_features.view(bs, -1)
        perts = self._make_perts((bs, C, H, W), rand_init=1)

        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            nowx = self._input_diversity(nowx)
            self.model.encode_image(nowx)
            
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = F.cosine_similarity(now_features.float(), ori_features.float()) 
            loss = torch.sum(this_loss)
            self._go(loss, perts)

        return self._get_adv(unnorm, perts)


class GLA(Base_all):
    '''
    Our methods, global augmentation and local augmentation
    '''
    def __init__(self, args):
        super(GLA, self).__init__(args)
        self.patch_size = args.patch_size
        self.range_left = args.range_left

    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)

        self.model.encode_image(normed_inps)
        ori_features = self._get_middle_feature()
        ori_features = ori_features.view(bs, -1)

        perts = self._make_perts((bs, C, H, W))
        upper_limit = ((1 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        lower_limit = ((0 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            valnow = int(self.patch_size)

            patch_transform_image = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(224, 224), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(valnow, valnow), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            nowx = patch_level_aug_here(nowx, patch_transform_image, upper_limit, lower_limit, 224)
            nowx = patch_level_aug_here(nowx, patch_transform, upper_limit, lower_limit, valnow)
            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = F.cosine_similarity(now_features.float(), ori_features.float()) 
            loss = torch.sum(this_loss)
            if self.l2reg:
                loss = loss - self.l2reg_alpha * torch.norm(perts)

            self._go(loss, perts)
        return self._get_adv(unnorm, perts)


class RandomMaxAttack(Base_all):
    '''
    random abs max noise
    '''
    def __init__(self, args):
        super(RandomMaxAttack, self).__init__(args)
    
    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        perts = torch.rand(bs, C, H, W)
        perts = torch.where(perts > 0.5, -self.epsilon, self.epsilon).to(self.device)
        nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
        return nowx.detach()


class rebuttal_l2norm_attck(Base_all):
    '''
    Using middle layer's features to calculate cos loss, without label information.
    '''
    def __init__(self, args):
        super(rebuttal_l2norm_attck, self).__init__(args)
        
    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        self.model.encode_image(normed_inps)         # forward for get ori_activation
        #ori_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
        ori_features = self._get_middle_feature()
        ori_features = ori_features.view(bs, -1)
        perts = self._make_perts((bs, C, H, W), rand_init=1)
        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = -torch.norm(now_features.float() - ori_features.float(), dim=-1) 
            loss = torch.sum(this_loss)
            self._go(loss, perts)
        return self._get_adv(unnorm, perts)

class rebuttal_GLA_l2norm_attack(Base_all):
    def __init__(self, args):
        super(rebuttal_GLA_l2norm_attack, self).__init__(args)
        self.patch_size = args.patch_size
        self.range_left = args.range_left

    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        self.model.encode_image(normed_inps)         # forward for get ori_activation
        #ori_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
        ori_features = self._get_middle_feature()
        ori_features = ori_features.view(bs, -1)
        perts = self._make_perts((bs, C, H, W))
        upper_limit = ((1 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        lower_limit = ((0 - self.mean) / self.std).view(3, 1, 1).to(self.device)

        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)

            valnow = int(self.patch_size)
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(valnow, valnow), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            patch_transform_image = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(224, 224), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            nowx = patch_level_aug_here(nowx, patch_transform_image, upper_limit, lower_limit, 224)
            nowx = patch_level_aug_here(nowx, patch_transform, upper_limit, lower_limit, valnow)

            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = -torch.norm(now_features.float() - ori_features.float(), dim=-1) 
            loss = torch.sum(this_loss)
            self._go(loss, perts)
        return self._get_adv(unnorm, perts)

class rebuttal_bigger_2norm(Base_all):
    '''
    Using middle layer's features to calculate cos loss, without label information.
    '''
    def __init__(self, args):
        super(rebuttal_bigger_2norm, self).__init__(args)
        
    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        perts = self._make_perts((bs, C, H, W), rand_init=1)

        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = -torch.norm(now_features.float(), dim=-1) 
            loss = torch.sum(this_loss)
            self._go(loss, perts)
        return self._get_adv(unnorm, perts)

class rebuttal_GLA_bigger_2norm(Base_all):
    '''
    Using middle layer's features to calculate cos loss, without label information.
    '''
    def __init__(self, args):
        super(rebuttal_GLA_bigger_2norm, self).__init__(args)
        self.patch_size = args.patch_size
        self.range_left = args.range_left

    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        perts = self._make_perts((bs, C, H, W))
        upper_limit = ((1 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        lower_limit = ((0 - self.mean) / self.std).view(3, 1, 1).to(self.device)

        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)

            valnow = int(self.patch_size)
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(valnow, valnow), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            patch_transform_image = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(224, 224), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            nowx = patch_level_aug_here(nowx, patch_transform_image, upper_limit, lower_limit, 224)
            nowx = patch_level_aug_here(nowx, patch_transform, upper_limit, lower_limit, valnow)



            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = -torch.norm(now_features.float(), dim=-1) 
            loss = torch.sum(this_loss)
            self._go(loss, perts)
        return self._get_adv(unnorm, perts)

class rebuttal_GLA_std(Base_all):
    '''
    Using middle layer's features to calculate cos loss, without label information.
    '''
    def __init__(self, args):
        super(rebuttal_GLA_std, self).__init__(args)
        self.patch_size = args.patch_size
        self.range_left = args.range_left
        
    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        perts = self._make_perts((bs, C, H, W))
        upper_limit = ((1 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        lower_limit = ((0 - self.mean) / self.std).view(3, 1, 1).to(self.device)

        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)

            valnow = int(self.patch_size)
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(valnow, valnow), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            patch_transform_image = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(224, 224), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            nowx = patch_level_aug_here(nowx, patch_transform_image, upper_limit, lower_limit, 224)
            nowx = patch_level_aug_here(nowx, patch_transform, upper_limit, lower_limit, valnow)



            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            loss = torch.sum(now_features.std(dim=-1))
            self._go(loss, perts)
        return self._get_adv(unnorm, perts)

class Multi_class_token_Cos_attack(Base_all):
    '''
    use muiti layer cos loss
    '''
    def __init__(self, args):
        super(Multi_class_token_Cos_attack, self).__init__(args)
        self._all_make_hook()

    def _get_class_token(self, name):
        def hook(model, input, output):
            self.features_hook[name] = output[0].unsqueeze(0)
        return hook
    
    def _get_other_token(self, name):
        def hook(model, input, output):
            self.features_hook[name] = output[1:]
        return hook

    def _all_make_hook(self):
        if self.model_name == "ViT-B/16" or self.model_name == 'ViT-B/32':
            for i in range(12):
                # if i <= 6:
                #     self.model.visual.transformer.resblocks[i].register_forward_hook(self._get_other_token("layer{0}".format(i)))
                # else:
                self.model.visual.transformer.resblocks[i].register_forward_hook(self._get_class_token("layer{0}".format(i)))

    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        self.model.encode_image(normed_inps)         # forward for get ori_activation
        ori_features = [self.features_hook["layer{0}".format(i)].permute(1, 0, 2).view(bs, -1) for i in range(12)]
        perts = self._make_perts((bs, C, H, W), rand_init=1)
        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            self.model.encode_image(nowx)
            now_features = [self.features_hook["layer{0}".format(i)].permute(1, 0, 2).view(bs, -1) for i in range(12)] 
            
            loss = 0
            for i in range(12):    
                loss = loss + torch.sum(F.cosine_similarity(now_features[i].float(), ori_features[i].float()))
            self._go(loss, perts)
        return self._get_adv(unnorm, perts)

class rebuttal_DA_ADAM(Base_all):
    def __init__(self, args):
        super(rebuttal_DA_ADAM, self).__init__(args)
    
    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)
        self.model.encode_image(normed_inps)         # forward for get ori_activation
        #ori_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
        ori_features = self._get_middle_feature()
        ori_features = ori_features.view(bs, -1)
        perts = self._make_perts((bs, C, H, W))
        g_a = 0
        g_t = 0
        for i in range(self.steps):
            for t in range(30):
                nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
                val = torch.empty(bs, C, H, W).normal_(0, 0.05).cuda()
                here = nowx + val
                here = self._norm(here)
                self.model.encode_image(here)
                now_features = self._get_middle_feature()
                now_features = now_features.view(bs, -1)   
                this_loss = F.cosine_similarity(now_features.float(), ori_features.float()) 
                loss = torch.sum(this_loss)
                loss.backward()
                g_a += perts.grad.data.sign()
                perts.grad.data.zero_()
            tmp = g_a.view(bs, -1)
            tmp = tmp / (tmp.norm(p=1, dim=-1).view(bs, 1))
            tmp = tmp.view(bs, C, H, W)
            g_t = 1.0 * g_t + tmp
            perts.data = perts.data - self.step_size * g_t.sign()
            perts.data  = torch.clamp(perts.data, -self.epsilon, self.epsilon)
        return self._get_adv(unnorm, perts)

class ablation_only_GA(Base_all):
    def __init__(self, args):
        super(ablation_only_GA, self).__init__(args)
        self.range_left = args.range_left

    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)

        self.model.encode_image(normed_inps)
        #ori_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
        ori_features = self._get_middle_feature()
        ori_features = ori_features.view(bs, -1)

        perts = self._make_perts((bs, C, H, W), rand_init=1)
        upper_limit = ((1 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        lower_limit = ((0 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            patch_transform_image = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(224, 224), scale=(self.range_left,1.0), ratio=(1.0,1.0), p=1), # 1
            )
            nowx = patch_level_aug_here(nowx, patch_transform_image, upper_limit, lower_limit, 224)
            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = F.cosine_similarity(now_features.float(), ori_features.float()) 
            loss = torch.sum(this_loss)
            if self.l2reg:
                loss = loss - self.l2reg_alpha * torch.norm(perts)

            self._go(loss, perts)
        return self._get_adv(unnorm, perts)

class ablation_only_LA(Base_all):
    def __init__(self, args):
        super(ablation_only_LA, self).__init__(args)
        self.patch_size = args.patch_size
        self.range_left = args.range_left

    def forward(self, normed_inps):
        normed_inps = normed_inps.to(self.device)
        bs, C, H, W = normed_inps.shape
        unnorm = self._unnorm(normed_inps)

        self.model.encode_image(normed_inps)
        #ori_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
        ori_features = self._get_middle_feature()
        ori_features = ori_features.view(bs, -1)

        # perts = self._make_perts((bs, C, H, W), rand_init=1)
        perts = self._make_perts((bs, C, H, W), rand_init=0)
        upper_limit = ((1 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        lower_limit = ((0 - self.mean) / self.std).view(3, 1, 1).to(self.device)
        for i in range(self.steps):
            nowx = torch.clamp(unnorm + torch.clamp(perts, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            nowx = self._norm(nowx)
            valnow = int(self.patch_size)
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(valnow, valnow), scale=(self.range_left, 1.0), ratio=(1.0,1.0), p=1), # 1
            )
            nowx = patch_level_aug_here(nowx, patch_transform, upper_limit, lower_limit, valnow)
            self.model.encode_image(nowx)
            #now_features = self.features_hook["layer{0}".format(self.activion_layer)].permute(1, 0, 2)   # LND -> NLD
            now_features = self._get_middle_feature()
            now_features = now_features.view(bs, -1)   
            this_loss = F.cosine_similarity(now_features.float(), ori_features.float()) 
            loss = torch.sum(this_loss)
            if self.l2reg:
                loss = loss - self.l2reg_alpha * torch.norm(perts)

            self._go(loss, perts)
        return self._get_adv(unnorm, perts)
