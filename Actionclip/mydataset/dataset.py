import torch.utils.data as data
import torchvision
from PIL import Image
import pandas as pd
from torchvision.transforms import InterpolationMode
import os
import numpy as np
import json
import torch


class Mydataset(data.Dataset):
    def __init__(self, data_path, num_frame, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.num_frame = num_frame
        self._parse_list()
        self.initialized = False

    def _load_image(self, path):
        return [Image.open(path).convert('RGB')]
    
    def _parse_list(self):
        tmp = os.listdir(self.data_path)
        tmp.sort()
        self.video_list = tmp
        with open("/share/home/lyq/GLAattack/Actionclip/fixed_keyid.json", "r") as fp:
            self.keyid = json.load(fp)
    
    def __getitem__(self, index):
        record = self.video_list[index]
        indices = [i for i in range(self.num_frame)]
        return self.get(record, indices)


    def get(self, record, indices):
        images = list()

        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                # seg_imgs = self._load_image(os.path.join(self.data_path, record, str(p) + ".jpg"))   # for init
                seg_imgs = self._load_image(os.path.join(self.data_path, record, str(p).zfill(2) + ".png"))
                #print(np.asarray(seg_imgs[0], dtype=np.int).shape)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(os.path.join(self.data_path, record, str(p).zfill(2) + ".png")))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
    
        process_data = self.transform(images)   #(batch * 3, h, w)
        class_id = self.keyid[record.split('-')[0].replace('@', ' ')]
        return process_data, class_id #record

    def __len__(self):
        return len(self.video_list)

class GroupScale(object):
    def __init__(self, size, interpolation=InterpolationMode.BICUBIC):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                rst = np.concatenate(img_group, axis=2)
                # plt.imshow(rst[:,:,3:6])
                # plt.show()
                return rst


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = self.mean * (tensor.size()[0]//len(self.mean))
        std = self.std * (tensor.size()[0]//len(self.std))
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        
        if len(tensor.size()) == 3:
            # for 3-D tensor (T*C, H, W)
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif len(tensor.size()) == 4:
            # for 4-D tensor (C, T, H, W)
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor


class GroupUnNorm(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = self.mean * (tensor.size()[0]//len(self.mean))
        std = self.std * (tensor.size()[0]//len(self.std))
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        
        if len(tensor.size()) == 3:
            # for 3-D tensor (T*C, H, W)
            tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
        elif len(tensor.size()) == 4:
            # for 4-D tensor (C, T, H, W)
            tensor.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
        return tensor