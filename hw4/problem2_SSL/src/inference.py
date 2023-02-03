import csv
import torch
import json
import os
import pandas as pd
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
import torchvision.models as models
from datetime import datetime
import argparse
import json
from PIL import Image
from model import ImgClassifier


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained deeplabv3_resetnet101 model."
    )
    parser.add_argument("--input", help="input csv path")
    parser.add_argument("--img_dir", help="images to be classified")
    parser.add_argument(
        "--output",
        help="Output csv path",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    return parser.parse_args()


def preprocess(label_csv_path):
    df = pd.read_csv(label_csv_path)
    filenames = df["filename"]
    labels = df["label"].unique()
    labels = sorted(labels)
    return labels, filenames


def inference(img_dir, input_csv, outpath, modelpath, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_tfm = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        ]
    )
    label_list, filenames = preprocess(input_csv)
    filename = "test_pred.csv"
    if outpath.endswith(".csv"):
        filename = outpath.split("/")[-1]
        outpath = "/".join(outpath.split("/")[:-1])

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # inference_set = InfDataset(img_dir, test_tfm)
    # data_loader = DataLoader(
    #     inference_set,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    # )
    model = ImgClassifier(device)
    model.load_state_dict(torch.load(modelpath, map_location=device)["model"])
    model.eval()

    result = []
    fnames = []

    for fname in tqdm((filenames)):
        img = Image.open(os.path.join(img_dir, fname))
        img = test_tfm(img).unsqueeze(0)
        logits = model(img.to(device))
        x = logits.argmax(dim=-1).squeeze(-1).cpu().detach().numpy()
        result.append(x)
        fnames.append(fname)
    # result = np.concatenate(result)
    # fnames = np.concatenate(fnames)
    # Generate your submission
    df = pd.DataFrame({"filename": fnames, "label": [label_list[r] for r in result]})
    df.to_csv(os.path.join(outpath, filename), index=True)


if __name__ == "__main__":
    args = get_args()
    inference(args.img_dir, args.input, args.output, args.model)
