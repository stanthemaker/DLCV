import numpy as np
import os
import torch
from torchvision.transforms import functional as F
import argparse
from PIL import Image
import torchvision.utils as vutils

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class UnNormalize(object):
    def __init__(self):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument(
        "--image_dir", "-p", type=str, default="", help="Experiment name"
    )

    return parser.parse_args()


def main(image_dir: str):
    path = image_dir
    files = sorted(
        [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
    )
    unorm = UnNormalize()
    for i in range(len(files)):
        im = F.to_tensor(Image.open(files[i]))
        im = unorm(im)
        vutils.save_image(im, os.path.join(path, f"{i}.png"))


if __name__ == "__main__":
    args = get_args()
    main(args.image_dir)
