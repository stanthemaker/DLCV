import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np

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


class ImgDataset(Dataset):
    def __init__(self, data_dir, imgsize=224):
        super(ImgDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted([p for p in os.listdir(data_dir)])
        self.transform = T.transforms.Compose(
            [
                T.Resize((imgsize, imgsize)),
                T.transforms.Lambda(check_rgb),
                T.transforms.ToTensor(),
                T.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.transform(img)

        if img.size(0) != 3:
            img = torch.cat((img, img, img))

        return img, fname
