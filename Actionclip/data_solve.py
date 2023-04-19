from tarfile import RECORDSIZE
import dataset
import json
import torchvision
import os
from torchvision.utils import save_image

input_mean = [0.48145466, 0.4578275, 0.40821073]
input_std = [0.26862954, 0.26130258, 0.27577711]
transform = torchvision.transforms.Compose([dataset.GroupScale(256),
                                             dataset.GroupCenterCrop(224),
                                             dataset.Stack(roll=False),
                                             dataset.ToTorchFormatTensor(div=True),
                                             dataset.GroupNormalize(input_mean,
                                                            input_std)])

val_data = dataset.Mydataset('/share/test/lyq/GLAattack/actionclip/vit-32-right-K', 8, transform)

unnorm = dataset.GroupUnNorm(input_mean,input_std)

print(len(val_data))

idkey = dict()

with open("keyid.json") as fp:
    keyid = json.load(fp)

for key, val in keyid.items():
    idkey[val] = key

target_path = '/share/test/lyq/GLAattack/actionclip/vit-32-right-K-png/'
for data, record in val_data:
    now = os.path.join(target_path, record)
    if not os.path.exists(now):
        os.makedirs(now)
    data_all = unnorm(data).view(8, 3, 224, 224)
    for i in range(8):
        save_image(data_all[i], os.path.join(now, str(i) + '.png'))