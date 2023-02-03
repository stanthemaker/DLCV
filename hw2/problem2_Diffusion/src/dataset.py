import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms as T
from torch import nn


class MNISTdataset(Dataset):
    def __init__(self, data_dir, dfname, img_size, flip=False):
        super(MNISTdataset, self).__init__()

        self.data_dir = data_dir
        df = pd.read_csv(dfname)
        self.labels = df["label"].tolist()
        self.files = sorted(
            [
                os.path.join(data_dir, p)
                for p in os.listdir(data_dir)
                if p in df["image_name"].tolist()
            ]
        )

        self.transform = T.Compose(
            [
                T.Resize(img_size),
                T.RandomHorizontalFlip() if flip else nn.Identity(),
                T.ToTensor(),
                T.Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]
        label = self.labels[idx]
        img = Image.open(fname)
        img = self.transform(img)

        return img, label


class InferenceDataset(Dataset):
    def __init__(self, data_dir, dfname, img_size, flip=False):
        super(InferenceDataset, self).__init__()

        self.data_dir = data_dir
        df = pd.read_csv(dfname)
        self.labels = df["label"].tolist()
        self.files = sorted(
            [
                os.path.join(data_dir, p)
                for p in os.listdir(data_dir)
                if p in df["image_name"].tolist()
            ]
        )

        self.transform = T.Compose(
            [
                T.Resize(img_size),
                T.RandomHorizontalFlip() if flip else nn.Identity(),
                T.ToTensor(),
                T.Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]
        label = self.labels[idx]
        img = Image.open(fname)
        img = self.transform(img)

        return img, label
