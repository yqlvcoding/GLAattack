import torch
from models.modeling import ImageClassifier, ClassificationHead, ImageClassifier
import clip_cls.clip as clip
import models.utils_cls as utils
import argparse
import os
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler
import shutil
from torchvision.utils import save_image
import json

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform):
        super().__init__(path, transform)

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }
        

def get_dataloader(dataset, is_train):
    dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch

def eval_single_dataset(image_classifier, dataset, args):
    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    dataloader = get_dataloader(dataset, is_train=False)
    batched_data = enumerate(dataloader)
    device = args.device

    target_path = '/share/home/lyq/wise-ft/imagenet1k/'
    id = dict()
    name2id = dict()
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)  # bs, 3, 244, 244
            y = data['labels'].to(device)
            if 'image_paths' in data:
                image_paths = data['image_paths']   # /share/common/ImageDatasets/imagenet_2012/val/n01530575/ILSVRC2012_val_00010999.JPEG
            

            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device) # bs 1
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            for i in range(len(y)):
                val = int(y[i])
                if val in id.keys():
                    continue
                if pred[i, 0] == y[i]:
                    
                    filename = image_paths[i].split('/')[-1].replace(".JPEG", ".png")
                    save_image(unnorm(x[i]), os.path.join(target_path, filename))
                    name2id[filename] = val
                    id[val] = 1
        
        with open('name2id.json', 'w') as json_file:
            json_file.write(json.dumps(name2id, ensure_ascii=False, indent=4))


        top1 = correct / n
        metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1
    
    print(metrics['top1'])
    return metrics

def unnorm(x):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    tmpx = torch.clone(x)
    tmpx.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tmpx

class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=4,
                 ):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.populate_test()

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        #return 'imagenet'
        return ''

def evaluate(image_classifier, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    dataset_name = args.eval_datasets
    
    print('Evaluating on', dataset_name)
    dataset = ImageNet(
        image_classifier.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4,
    )
    eval_single_dataset(image_classifier, dataset, args)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-location", type=str, default='/share/common/ImageDatasets/imagenet_2012/')
    parser.add_argument("--eval-datasets", default='ImageNet', type=lambda x: x.split(","))
    parser.add_argument("--results-db", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model", type=str, default='ViT-B/32', help="For init structure, not weight")
    parser.add_argument("--freeze-encoder", default=False, action="store_true")
    parser.add_argument("--load", type=str, default='/share/home/lyq/wise-ft/myfinetunemodel/wise_ft_alpha=0.700.pt')
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args

class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False, pretrained=False)

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
