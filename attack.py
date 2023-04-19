import torch
from torch.utils.data import DataLoader
import os
import methods
from utils import set_random_seed, regain
from tqdm import tqdm
import argparse
import dataset
from torchvision.utils import save_image


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--attack', type=str, default='')
    parser.add_argument('--generate_model_name', type=str, default='ViT-B/16')
    parser.add_argument('--downstream_task', type=str, default='Actionclip', choices= ["Actionclip" , "CLIP2Video", "ImagenetCLS"])
    parser.add_argument('--clean_datapath', type=str, default='/share/test/lyq/GLAattack/actionclip/vit-32-right-K-png')
    parser.add_argument('--results_path', type=str, default='/share/test/lyq/GLAattack/actionclip/adv-vit-32-K')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--cpu_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--mean', type=float, default=None)
    parser.add_argument('--std', type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=16/255)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--init_adam_lr', type=int, default=0.005, help='do not change here, change in cmd parameters')
    parser.add_argument('--multi_step_size', type=float, default=None)
    parser.add_argument('--mid_layer', type=int, default=None, help='middle layer activations for calculate loss')
    parser.add_argument('--use_adam', type=int, default=0, help="0 means sgd, others(like 1) means using adam")
    parser.add_argument('--file_p', type=str, default=None)
    parser.add_argument('--name_last', type=str, default=None)
    parser.add_argument('--l2reg', type=str, default=0)
    parser.add_argument('--l2reg_alpha', type=float, default=0.1, help='not change here, change in cmd parameters')
    parser.add_argument('--no_save', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--range_left', type=float, default=0.85)
    parser.add_argument('--overwrite', type=int, default=0, help='regenerate adversarial examples')

    args = parser.parse_args()
    args.opt_path = os.path.join(args.results_path, '{}#{}'.format(args.generate_model_name.replace('/', '-'), args.attack))
    if args.use_adam:
        args.opt_path = args.opt_path + '_adam'

    flag = 0
    if args.file_p is not None:
        if flag == 0:
            midchar = '#'
        else:
            midchar = '_'
        args.opt_path = args.opt_path + midchar + 'PR{}'.format(args.file_p)
        flag = 1

    if args.mid_layer is not None:
        if flag == 0:
            midchar = '#'
        else:
            midchar = '_'
        args.opt_path = args.opt_path + midchar + 'cos-layer{}'.format(args.mid_layer)
        flag = 1
    
    if args.name_last is not None:
        if flag == 0:
            midchar = '#'
        else:
            midchar = '_'
        args.opt_path = args.opt_path + midchar + args.name_last
        flag = 1

    if not args.no_save and not os.path.exists(args.opt_path):
        os.makedirs(args.opt_path)
    return args


if __name__=="__main__":
    set_random_seed(1024)
    args = arg_parse()
    args.device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    print(args.device)
    
    ###### load data
    if args.downstream_task == "ImagenetCLS":
        val_data = dataset.MyDataset_imagenetCLS(args.clean_datapath)
        
    else:
        transform_val = dataset.get_transform()
        val_data = dataset.Mydataset(args.clean_datapath, transform_val)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.cpu_num, shuffle=False, pin_memory=True)
        
    
    attack_method = getattr(methods, args.attack)(args)

    
    print("run ", args.attack)
    for iii, (images, records) in enumerate(val_loader):
        if args.overwrite and args.downstream_task != "ImagenetCLS":
            images, records = regain(images, records, args)

        bs = images.shape[0]
        if bs == 0:
            continue
        
        assert(bs == len(records))
        
        if args.downstream_task != "ImagenetCLS":
            images = images.view((bs, -1, 3) + images.size()[-2:])  #batch, frame * c, h, w -> batch, frame, c, h, w
            b, t, c, h, w = images.size()
            image_input = images.to(args.device).view(-1, c, h, w)  # batch * frame, c, h, w
        else:
            image_input = images

        adv_inps = attack_method(image_input)

        assert(len(records) == bs)

        if args.downstream_task != "ImagenetCLS":
            sp_adv_inps = adv_inps.view(b, t, c, h, w)
            for i in range(b):
                save_path = os.path.join(args.opt_path, records[i])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for j in range(t):
                    save_image(sp_adv_inps[i][j], os.path.join(save_path, str(j).zfill(2) + '.png'))
        else:
            if not os.path.exists(args.opt_path):
                os.makedirs(args.opt_path)
            for i in range(bs):
                save_image(adv_inps[i], os.path.join(args.opt_path, records[i]))
