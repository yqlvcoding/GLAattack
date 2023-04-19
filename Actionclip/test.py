# Modified from Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"

from ast import parse
import os
import target_clip
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from mydataset.dataset import Mydataset
import argparse
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
from utils.Text_Prompt import *
import pandas as pd
import torch
from pandas import DataFrame


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug):
    model.eval()
    fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        for iii, (image, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    #print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    return top1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='/share/home/lyq/GLAattack/Actionclip/configs/v32_8f_k400_test.yaml')
    parser.add_argument('--data_path_name', type=str)
    parser.add_argument('--num_frame', type=int, default=8)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--target_model', type=str, default='ac-vit32-8f')
    parser.add_argument('--change_comment', type=str, default=None)
    args = parser.parse_args()
    args.data_path = os.path.join('/share/test/lyq/GLAattack/actionclip/adv-vit-32-K', args.data_path_name)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = target_clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation()

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    
    transform = get_augmentation()
    val_data = Mydataset(args.data_path, args.num_frame, transform)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=False)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        target_clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        target_clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))
    

    classes_all = pd.read_csv(config.data.label_list)
    classes, num_text_aug, text_dict = text_prompt(classes_all.values.tolist())

    source_model = args.data_path_name.split('#')[0]
    target_model = args.target_model
    tmp = args.data_path_name.split('#', 1)[1].split('#', 1)
    comment = '-'
    if len(tmp) == 1:
        method = tmp[0]
    else:
        method = tmp[0]
        comment = tmp[1]
    
    if args.change_comment is not None:
        comment = args.change_comment



    prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)
    if not args.save:
        exit(0)
    
    
    if os.path.exists("/share/home/lyq/GLAattack/Actionclip/Actionclip_attack_results.csv"):
        
        csv_df = pd.read_csv("Actionclip_attack_results.csv")
        res = csv_df.set_index(['SOURCE', 'METHOD', 'TARGET', 'COMMENT']).T.to_dict()
        if (source_model, method, target_model, comment) in res.keys():
            print("record already exists")
        else:
            row_old = csv_df.shape[0]
            data = {
                'COL': row_old, 
                'TARGET': target_model,
                'SOURCE': source_model,
                'METHOD': method,
                'ASR': 100 - prec1, 
                'ACC': prec1, 
                'COMMENT': comment
            }
            df = DataFrame(data, columns= ['COL', 'TARGET', 'SOURCE', 'METHOD', 'ASR', 'ACC', 'COMMENT'], index=[0])
            df.to_csv('/share/home/lyq/GLAattack/Actionclip/Actionclip_attack_results.csv', mode='a', header=None, index=None, float_format='%.2f')
    else:
        data = {
            'COL': 0, 
            'TARGET': target_model,
            'SOURCE': source_model,
            'METHOD': method,
            'ASR': 100 - prec1, 
            'ACC': prec1, 
            'COMMENT': comment
        }
        df = DataFrame(data, columns= ['COL', 'TARGET', 'SOURCE', 'METHOD', 'ASR', 'ACC', 'COMMENT'], index=[0])
        df.to_csv('/share/home/lyq/GLAattack/Actionclip/Actionclip_attack_results.csv', mode='a', index=None, float_format='%.2f')

    
    
if __name__ == '__main__':
    main()



#######   Top1: 71.10995850622407, Top5: 90.79875518672199

#######   单纯clip  Top1: 23.210580912863072, Top5: 47.78008298755187