from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd


def svhn_tfm(image, image_size):
    train_tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = TF.resize(image, image_size)

    # image = TF.to_tensor(image)
    image = train_tfm(image)
    return image


def usps_tfm(image, image_size):
    train_tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    image = TF.resize(image, image_size)
    # image = TF.to_tensor(image)
    image = train_tfm(image)

    # image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image


class SVHNdataset:
    def __init__(self, datapath, train: bool, source: bool):
        # self.files =

        self.is_source = source
        self.is_train = train
        if self.is_train:
            sort_csv = pd.read_csv(os.path.join(datapath, "train.csv")).sort_values(
                "image_name"
            )
        else:
            sort_csv = pd.read_csv(os.path.join(datapath, "val.csv")).sort_values(
                "image_name"
            )
        self.labels_list = sort_csv.label.values[:]
        print(f"finish building label list at {datapath} Training:{self.is_train}")
        self.filenames = sort_csv.image_name.values[:]
        self.images_list = sorted(
            [
                os.path.join(datapath, "data", x)
                for x in os.listdir(os.path.join(datapath, "data"))
                if x in self.filenames
            ]
        )
        print(
            f"finish building image list at {datapath}, number of images:{len(self.images_list)}"
        )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        im = svhn_tfm(Image.open(self.images_list[idx]), 28, aug=(self.is_train))

        # source train and valid / target valid -> image and label
        label = self.labels_list[idx]

        # target train -> only image

        return im, label


class USPSdataset:
    def __init__(self, datapath, train: bool, source: bool):
        # self.files =

        self.is_source = source
        self.is_train = train
        if self.is_train:
            sort_csv = pd.read_csv(os.path.join(datapath, "train.csv")).sort_values(
                "image_name"
            )
        else:
            sort_csv = pd.read_csv(os.path.join(datapath, "val.csv")).sort_values(
                "image_name"
            )
        self.labels_list = sort_csv.label.values[:]
        print(f"finish building label list at {datapath} Training:{self.is_train}")
        self.filenames = sort_csv.image_name.values[:]
        self.images_list = sorted(
            [
                os.path.join(datapath, "data", x)
                for x in os.listdir(os.path.join(datapath, "data"))
                if x in self.filenames
            ]
        )
        print(
            f"finish building image list at {datapath}, number of images:{len(self.images_list)}"
        )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        im = usps_tfm(Image.open(self.images_list[idx]), 28, aug=(self.is_train))

        # source train and valid / target valid -> image and label
        label = self.labels_list[idx]

        # target train -> only image

        return im, label


class InfSVHNdataset:
    def __init__(self, datapath):
        # self.files =
        self.images_list = sorted(
            [os.path.join(datapath, x) for x in os.listdir(datapath)]
        )
        print(
            f"finish building image list at {datapath}, number of images:{len(self.images_list)}"
        )
        self.filenames = sorted([x for x in os.listdir(datapath)])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        im = svhn_tfm(Image.open(self.images_list[idx]), 28)
        return im

    def get_filelist(self):
        return self.filenames


class InfUSPSdataset:
    def __init__(self, datapath):
        # self.files =
        self.images_list = sorted(
            [os.path.join(datapath, x) for x in os.listdir(datapath)]
        )
        print(
            f"finish building image list at {datapath}, number of images:{len(self.images_list)}"
        )
        self.filenames = sorted([x for x in os.listdir(datapath)])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        im = usps_tfm(Image.open(self.images_list[idx]), 28)
        return im

    def get_filelist(self):
        return self.filenames
