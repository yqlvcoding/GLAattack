import torch
from models.modeling import ImageClassifier, ClassificationHead, ImageClassifier
import clip_cls.clip as clip
import models.utils_cls as utils
import argparse
import os
import pandas as pd
from torch.utils.data import DataLoader

from ourdataset import OurDataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-path', type=str, default='/share/test/lyq/GLAattack/imagenetCLS/abalation_adv')
    parser.add_argument("--data-path-name", type=str)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--cpu-num", type=int, default=4)
    parser.add_argument("--change-comment", type=str, default=None)
    parser.add_argument("--target-model", type=str, default='ViT-B/32')
    parser.add_argument("--load", type=str, default='/share/home/lyq/wise-ft/myfinetunemodel/wise_ft_alpha=0.700.pt')
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


def eval(model, dataloader):
    batched_data = enumerate(dataloader)
    device = args.device


    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, (images, labels, names) in batched_data:
            images = images.to(device)  # bs, 3, 244, 244
            labels = labels.to(device)
            logits = utils.get_logits(images, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device) # bs 1
            correct += pred.eq(labels.view_as(pred)).sum().item()
            n += labels.size(0)

        top1 = correct / n
        metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1
    return metrics

def evaluate(image_classifier, args):
    dataset = OurDataset(os.path.join(args.target_path, args.data_path_name))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.cpu_num)
    results = eval(image_classifier, data_loader)

    # if 'top1' in results:
    #     print(f"Top-1 accuracy: {results['top1']:.4f}")
    
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
    
    prec1 = results['top1'] * 100
    if os.path.exists("/share/home/lyq/GLAattack/ImagenetCLS/imagenetclip_attack_results.csv"):
        csv_df = pd.read_csv("imagenetclip_attack_results.csv")
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
            df = pd.DataFrame(data, columns= ['COL', 'TARGET', 'SOURCE', 'METHOD', 'ASR', 'ACC', 'COMMENT'], index=[0])
            df.to_csv('/share/home/lyq/GLAattack/ImagenetCLS/imagenetclip_attack_results.csv', mode='a', header=None, index=None, float_format='%.2f')
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
        df = pd.DataFrame(data, columns= ['COL', 'TARGET', 'SOURCE', 'METHOD', 'ASR', 'ACC', 'COMMENT'], index=[0])
        df.to_csv('/share/home/lyq/GLAattack/ImagenetCLS/imagenetclip_attack_results.csv', mode='a', index=None, float_format='%.2f')


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.target_model, args.device, jit=False, pretrained=False)

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


def make_model(args):
    image_encoder = ImageEncoder(args, keep_lang=True)
    classification_head = ClassificationHead(normalize=True, weights=torch.rand(1000, 512))
    delattr(image_encoder.model, 'transformer')
    new_model = ImageClassifier(image_encoder, classification_head, process_images=True)
    return new_model

def run_eval(args):
    merged_checkpoint = args.load
    merged = make_model(args)
    merged.load_state_dict(torch.load(merged_checkpoint))
    evaluate(merged, args)


if __name__ == '__main__':
    args = parse_arguments()
    run_eval(args)
