import os
import clip
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import pandas as pd
import argparse
import random


# Load the model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_dir",
    "-i",
    type=str,
    default="/home/stan/hw3-stanthemaker/hw3_data/p1_data/val",
    help="images path",
)
parser.add_argument(
    "--json_path",
    "-j",
    type=str,
    default="/home/stan/hw3-stanthemaker/hw3_data/p1_data/id2label.json",
    help="json path",
)
parser.add_argument(
    "--csv_path",
    "-c",
    type=str,
    default="/home/stan/hw3-stanthemaker/problem1_Evaluate/output/pred.csv",
    help="output csv path",
)
parser.add_argument(
    "--cuda",
    type=str,
    default="cuda",
    help="choose a cuda",
)
args = parser.parse_args()


class InfDataset(Dataset):
    def __init__(self, path):
        super(Dataset).__init__()
        self.path = path
        self.files = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
        )
        self.filenames = sorted(
            [file for file in os.listdir(path) if file.endswith(".png")]
        )
        # self.files = random.sample(self.files, 3)
        print(f"One {path} sample", self.files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = preprocess(im)
        fname = fname.split("/")[-1]
        # print(fname)
        # label = int(fname.split("_")[0])
        return im


device = args.cuda if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
dataset = InfDataset(path=args.image_dir)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

json_data = []
with open(args.json_path) as json_file:
    json_data = json.load(json_file)
data_classes = list(json_data.values())

acc_count = 0
progress_bar = tqdm(dataloader)
progress_bar.set_description(f"Inferencing...")
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data_classes]).to(
    device
)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

pred_labels = []
for i, imgs in enumerate(progress_bar):
    imgs = imgs.to(device)

    with torch.no_grad():
        image_features = model.encode_image(imgs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    _, pred = similarity[0].topk(1)
    pred = pred[0].cpu().detach().numpy()
    pred_labels.append(pred)

df = pd.DataFrame({"filename": dataset.filenames, "label": pred_labels})
df.to_csv(args.csv_path, index=False)
