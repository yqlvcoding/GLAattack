import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import datetime

def print_time():
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time)


def patch_level_aug_here(input1, patch_transform, upper_limit, lower_limit, p_size, nums_2):
    B, C, H, W = input1.shape
    patches = input1.unfold(2, p_size, p_size).unfold(3, p_size, p_size).permute(0,2,3,1,4,5).contiguous().reshape(B, -1, C, p_size, p_size)
    patches = patch_transform(patches.reshape(-1, C, p_size, p_size)).reshape(B, -1, C, p_size, p_size)
    patches = patches.reshape(B, -1, C, p_size, p_size).permute(0,2,3,4,1).contiguous().reshape(B, C * p_size * p_size, -1)  # (128, 3 * 16 * 16, 196)
    output_images = F.fold(patches, (H,W), p_size, stride=p_size)  # b * 3 * 224 * 224
    output_images = torch.max(torch.min(output_images, upper_limit), lower_limit)
    return output_images


def set_random_seed(seedno=1024):
    random.seed(seedno)
    np.random.seed(seedno)
    torch.manual_seed(seedno)
    torch.cuda.manual_seed(seedno)


def regain(image, records, args):
    b = image.shape[0]
    assert(b == len(records))
    indics = list()
    new_records = list()
    for i in range(b):
        save_path = os.path.join(args.opt_path, records[i])
        if os.path.exists(save_path):
            if args.downstream_task == 'CLIP2Video' and len(os.listdir(save_path)) == 12:
                continue
            elif args.downstream_task == 'Actionclip' and len(os.listdir(save_path)) == 8:
                continue
        indics.append(i)
        new_records.append(records[i])
    return image[indics, ...], new_records
    
