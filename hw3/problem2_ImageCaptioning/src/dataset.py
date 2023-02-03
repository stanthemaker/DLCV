import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
from PIL import Image
import json
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

## max len of train: 54
## max len of val : 50
MAX_DIM = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def under_max(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


def check_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


class collate_fn(object):
    def __call__(self, batch):
        """
        Padds batch of variable lengthes to a fixed length (max_len)
        """
        imgs, img_masks, captions, cap_masks, names = zip(*batch)
        imgs = torch.stack(imgs)  # (B, 3, 256, 256)
        img_masks = torch.stack(img_masks)
        captions = torch.stack(captions)
        cap_masks = torch.stack(cap_masks)

        return imgs, img_masks, captions, cap_masks, names


class ImgCaptrionDataset(Dataset):
    def __init__(self, data_dir, tokenizer, caption_path, imgsize=224, train=True):
        super(ImgCaptrionDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted([p for p in os.listdir(data_dir)])
        # self.transform = T.Compose([
        #                 T.Resize((imgsize, imgsize)),
        #                 T.ToTensor()
        #                 ])
        train_transform = T.Compose(
            [
                T.Resize((imgsize, imgsize)),
                RandomRotation(),
                T.Lambda(check_rgb),
                T.ColorJitter(
                    brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )

        val_transform = T.Compose(
            [
                T.Resize((imgsize, imgsize)),
                T.Lambda(check_rgb),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )

        if train:
            self.transform = train_transform
        else:
            self.transform = val_transform
        self.tokenizer = tokenizer

        with open(caption_path, "r") as caption_file:
            self.caption_dict = json.load(caption_file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.transform(img)
        # print(img.size())

        id = [
            dic["id"]
            for dic in self.caption_dict["images"]
            if dic["file_name"] == fname
        ]
        id = id[0]
        caption_ls = [
            dic["caption"]
            for dic in self.caption_dict["annotations"]
            if dic["image_id"] == id
        ]
        caption_ls = [self.tokenizer.encode(caption) for caption in caption_ls]
        captions = [
            torch.as_tensor(caption.ids, dtype=torch.long) for caption in caption_ls
        ]
        captions = pad_sequence(captions, batch_first=True, padding_value=0)
        if captions.size(0) == 6:
            # print(id)
            captions = captions[1:, :]

        if img.size(0) != 3:
            img = torch.cat((img, img, img))

        name = fname.split(".")[0]

        return img, name, torch.transpose(captions, 0, 1)


class collate_padd(object):
    def __init__(self, max_len=54, pad_id=0):
        self.pad = pad_id
        self.max_len = max_len

    def __call__(self, batch):
        """
        Padds batch of variable lengthes to a fixed length (max_len)
        """
        imgs, names, captions = zip(*batch)
        # imgs, names = zip(*batch)
        imgs = torch.stack(imgs)  # (B, 3, 256, 256)
        # for cap in captions:
        #     print(cap.size())
        captions = pad_sequence(
            captions, batch_first=True, padding_value=self.pad
        )  # (B, max_len, captns_num=5)

        pad_right = self.max_len - captions.size(1)
        if pad_right > 0:
            # [B, captns_num, max_seq_len]
            captions = captions.permute(0, 2, 1)
            captions = nn.ConstantPad1d((0, pad_right), value=self.pad)(captions)
            captions = captions.permute(0, 2, 1)

        return imgs, names, captions


class Infcollate_padd(object):
    def __init__(self, max_len=54, pad_id=0):
        self.pad = pad_id
        self.max_len = max_len

    def __call__(self, batch):
        """
        Padds batch of variable lengthes to a fixed length (max_len)
        """
        imgs, names = zip(*batch)
        imgs = torch.stack(imgs)  # (B, 3, 256, 256)
        return imgs, names


class InfDataset(Dataset):
    def __init__(self, data_dir, imgsize=384):
        super(InfDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted([p for p in os.listdir(data_dir)])
        val_transform = T.Compose(
            [
                T.Resize((imgsize, imgsize)),
                T.Lambda(check_rgb),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )
        self.transform = val_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.transform(img)
        if img.size(0) != 3:
            img = torch.cat((img, img, img))

        name = fname.split(".")[0]

        return img, name
