import torch.utils.data as data
import torchvision
from PIL import Image
import pandas as pd
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os
import numpy as np
import json
import torch
import glob


class Mydataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self._parse_list()
        self.initialized = False

    def _load_image(self, path):
        return [Image.open(path).convert('RGB')]
    
    def _parse_list(self):
        tmp = os.listdir(self.data_path)
        tmp.sort()
        self.video_list = tmp
        

    def __getitem__(self, index):
        record = self.video_list[index]
        return self.get(record)


    def get(self, record):
        images = list()
        now_path = os.path.join(self.data_path, record)
        files = os.listdir(now_path)
        files.sort()

        for filename in files:
            try:
                seg_imgs = self._load_image(os.path.join(now_path, filename))
            except OSError:
                print('ERROR: Could not read image "{}"'.format(os.path.join(now_path, filename)))
                raise
            images.extend(seg_imgs)
    
        process_data = self.transform(images)
        return process_data, record

    def __len__(self):
        return len(self.video_list)


def nowtransforms_imagenetCLS():
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
    return transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])


class MyDataset_imagenetCLS(data.Dataset):
    def __init__(self, data_path):
        self.transform = nowtransforms_imagenetCLS()
        paths = glob.glob(os.path.join(data_path, '*.png'))
        paths = [i.split('/')[-1] for i in paths]
        paths = [i.strip() for i in paths]
        self.paths = [os.path.join(data_path, fname) for fname in paths]
        
        with open("ImagenetCLS/name2id.json", "r") as fp:
            self.json_info = json.load(fp)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image_name = path.split('/')[-1]
        class_id = self.json_info[image_name]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, image_name


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
    

def get_transform():
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)])
    return transform