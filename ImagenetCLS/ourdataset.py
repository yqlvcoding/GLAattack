import torch.utils.data as data
from torchvision import transforms
import glob
import os
import json
from PIL import Image

def nowtransforms():
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
    return transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])


class OurDataset(data.Dataset):
    def __init__(self, data_path):
        self.transform = nowtransforms()
        paths = glob.glob(os.path.join(data_path, '*.png'))
        paths = [i.split('/')[-1] for i in paths]
        paths = [i.strip() for i in paths]
        self.paths = [os.path.join(data_path, fname) for fname in paths]
        
        with open("name2id.json", "r") as fp:
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
        return img, class_id, image_name