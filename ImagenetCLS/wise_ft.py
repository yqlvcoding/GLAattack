import os

import numpy as np

import torch
import clip_cls.clip as clip
import models.utils_cls as utils
from models.eval import evaluate
from models.finetune import finetune
from models.modeling import ClassificationHead, ImageClassifier
from models.utils_cls import fisher_load
from models.zeroshot import get_zeroshot_classifier
from args import parse_arguments



class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False, pretrained=False)
        
        self.cache_dir = args.cache_dir

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

def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta

def make_model(args):
    ##### added
    image_encoder = ImageEncoder(args, keep_lang=True)
    classification_head = ClassificationHead(normalize=True, weights=torch.rand(1000, 512))
    delattr(image_encoder.model, 'transformer')
    new_model = ImageClassifier(image_encoder, classification_head, process_images=True)
    return new_model
    ##### added

def wise_ft(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    if args.load is None:
        print("######### finetune firstly #############")
        # Build and save zero-shot model
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
        
        zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
        classifier.save(zeroshot_checkpoint)

        # Standard fine-tuning
        args.load = zeroshot_checkpoint
        args.save = os.path.join(args.save, 'finetuned')
        finetuned_checkpoint = finetune(args)
    else:
        # No need to compute things from stratch
        assert len(args.load) == 2
        zeroshot_checkpoint, finetuned_checkpoint = args.load

    # Load model_dict
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)

    theta_0 = {k: v.clone() for k, v in zeroshot.items()}
    theta_1 = {k: v.clone() for k, v in finetuned.items()}
    del zeroshot

    if args.fisher is None:
        fishers = None
    else:
        fisher_0_file, fisher_1_file = args.fisher
        fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
        fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
        fishers = fisher_0, fisher_1

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

        new_model = make_model(args)
        # update the model (in-place) acccording to the new weights
        
        new_model.load_state_dict(theta)

        # save model
        new_model.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(new_model, args)


if __name__ == '__main__':
    args = parse_arguments()
    wise_ft(args)
