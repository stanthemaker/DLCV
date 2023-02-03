import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from utils import CIFAR10Policy
from PIL import Image


# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
train_tfm = transforms.Compose(
    [
        CIFAR10Policy(),
        # transforms.Resize(128),
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_tfm = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])


class InfDataset(Dataset):
    def __init__(self, path, tfm=test_tfm):
        super(Dataset).__init__()
        self.path = path
        
        self.files = sorted([os.path.join(path, x)
                            for x in os.listdir(path) if x.endswith(".png")])
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


class VehilcleDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None, mode="train"):
        super(VehilcleDataset).__init__()
        self.path = path
        self.files = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
        )
        self.mode = mode
        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        # type = <class 'torchvision.transforms.transforms.Compose'>
        self.transform_train = train_tfm
        self.transform_test = test_tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        length = self.__len__()
        fname = self.files[idx]
        im = Image.open(fname)
        if self.mode == "train":
            im = self.transform_train(im)

        else:
            im = self.transform_test(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label
