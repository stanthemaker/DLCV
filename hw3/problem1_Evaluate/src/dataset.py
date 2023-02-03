from PIL import Image
from torch.utils.data import DataLoader, Dataset
import clip
import os


class InfDataset(Dataset):
    def __init__(self, path):
        super(Dataset).__init__()
        self.path = path
        self.files = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
        )
        print(f"One {path} sample", self.files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        label = fname.split("/")[-1].split("_")[0]
        label = int(label)
        im = Image.open(fname)
        im = preprocess(im)
        return im, label
