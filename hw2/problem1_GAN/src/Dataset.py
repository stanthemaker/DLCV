import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torchvision.transforms import functional as TF

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
train_tfm = transforms.Compose(
    [
        transforms.RandomResizedCrop((64, 64), (0.8, 1.25), (0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


class UnNormalize(object):
    def __init__(self):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class InfDataset(Dataset):
    def __init__(self, path, tfm):
        super(Dataset).__init__()
        self.path = path

        self.files = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
        )
        self.filenames = [file for file in os.listdir(path)]
        self.filenames.sort()
        self.transform = tfm
        print(f"One {path} sample", self.files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        return im


class PhotoDataset(Dataset):
    def __init__(self, path, tfm=train_tfm, mode="train"):
        super(PhotoDataset).__init__()
        self.path = path
        self.files = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
        )
        self.mode = mode
        print(f"One {path} sample", self.files[0])
        # type = <class 'torchvision.transforms.transforms.Compose'>
        self.transform_train = train_tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform_train(im)
        return im


# im = Image.open("/home/stan/hw2-stanthemaker/hw2_data/face/val/38464.png")
# img = TF.to_tensor(im)
# print(img.shape)
