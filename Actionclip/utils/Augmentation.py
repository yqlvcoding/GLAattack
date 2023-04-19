# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

from target_clip.model import Transformer
from utils.transforms_ss import *

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

def get_augmentation():
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)])
    return transform



def unnorm(x):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    mean = input_mean * (x.size()[0]//len(input_mean))
    std = input_std * (x.size()[0]//len(input_std))
    mean = torch.Tensor(mean).cuda()
    std = torch.Tensor(std).cuda()
    
    if len(x.size()) == 3:
        # for 3-D tensor (T*C, H, W)
        x.mul_(std[:, None, None]).add_(mean[:, None, None])
    elif len(x.size()) == 4:
        # for 4-D tensor (C, T, H, W)
        x.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
    return x

def norm(x):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    mean = input_mean * (x.size()[0]//len(input_mean))
    std = input_std * (x.size()[0]//len(input_std))
    mean = torch.Tensor(mean).cuda()
    std = torch.Tensor(std).cuda()
    if len(x.size()) == 3:
        # for 3-D tensor (T*C, H, W)
        x.sub_(mean[:, None, None]).div_(std[:, None, None])
    elif len(x.size()) == 4:
        # for 4-D tensor (C, T, H, W)
        x.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return x
