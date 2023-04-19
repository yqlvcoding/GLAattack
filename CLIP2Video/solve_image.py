import os
from PIL import Image
from torchvision.utils import save_image

with open("/share/home/lyq/GLAattack/CLIP2Video/data/msvd_data/correct_test_list.txt", 'r') as fp:
    video_ids = [itm.strip() for itm in fp.readlines()]
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

origin_path = '/share/test/lyq/GLAattack/clip2video/origin'
target_path = '/share/test/lyq/GLAattack/clip2video/correct'
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
    ])

preprocess = _transform(224)

for video_id in video_ids:
    path_now = os.path.join(origin_path, video_id)
    video_name = os.listdir(path_now)
    video_name.sort()
    target_path_now = os.path.join(target_path, video_id)
    if not os.path.exists(target_path_now):
        os.makedirs(target_path_now)
    for filename in video_name:
        image_path = os.path.join(path_now, filename)
        a = preprocess(Image.open(image_path).convert("RGB"))
        target_file = os.path.join(target_path_now, filename)
        save_image(a, target_file)