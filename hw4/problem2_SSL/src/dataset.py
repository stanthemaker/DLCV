from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms as T
import pandas as pd


class ImgDataset(Dataset):
    def __init__(self, data_dir):
        super(ImgDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted([p for p in os.listdir(data_dir)])
        self.tfm = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.tfm(img)
        return img


class FineTuneDataset(Dataset):
    def __init__(self, data_dir, label_dict, tfm):
        super(FineTuneDataset, self).__init__()
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.files = sorted([p for p in os.listdir(data_dir)])
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.tfm(img)

        label = fname.split(".")[0]
        label = label[:-5]
        if label != "None":
            label = self.label_dict[label]
        else:
            label = None
        return img, label


class InfDataset(Dataset):
    def __init__(self, data_dir, tfm):
        super(InfDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted([p for p in os.listdir(data_dir)])
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.tfm(img)
        return img, fname
