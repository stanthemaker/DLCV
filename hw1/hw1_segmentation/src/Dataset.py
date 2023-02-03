import os
from os.path import join
from pathlib import Path
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
import random
import torch
import imageio
from PIL import Image


def read_masks(file_list: list):
    """
    Read masks from directory and tranform to categorical
    """
    n_masks = len(file_list)
    masks = torch.empty((n_masks, 512, 512), dtype=torch.int)
    print("reading masks...")
    for i, file in enumerate(file_list):
        mask = imageio.imread(file)
        mask = mask > 128
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3  # (Green: 010) Forest land
        masks[i, mask == 1] = 4  # (Blue: 001) Water
        masks[i, mask == 7] = 5  # (White: 111) Barren land
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
    return masks

def transform(image: Image, mask: torch.Tensor ,aug:bool):
        mask  = TF.to_pil_image(mask)

        if random.random() > 0.5 and aug:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5 and aug:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, torch.squeeze(mask)

class groundDataset(Dataset):
    def __init__(
        self,
        path: str,
        mode: str,
    ):
        self.images_list = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")]
        )

        labels_list = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
        )
        self.mode = mode
        print("reading masks...")
        self.labels = read_masks(labels_list)
        

    def __len__(self):
        return len(self.images_list)
    

    def __getitem__(self, idx):
        sats , masks = transform(Image.open(self.images_list[idx]), self.labels[idx] , aug = (self.mode == "train"))

        return sats, masks