from net import Inceptionv3
from Dataset import InfDataset
import sys
import torch
import argparse
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained deeplabv3_resetnet101 model."
    )
    parser.add_argument(
        "--input", "-t", type=str, default=None, help="Load test data for evaluation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory path for predition masks",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    return parser.parse_args()


def inference(datapath, outpath, modelpath, batch_size=8):

    inference_set = InfDataset(datapath)
    inference_loader = DataLoader(
        inference_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # make output dir
    filename = "pred.csv"
    if outpath.endswith(".csv"):
        filename = outpath.split("/")[-1]
        outpath = "/".join(outpath.split("/")[:-1])

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Inceptionv3()
    model = model.to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()

    result = []

    for inference_data in inference_loader:
        inference_data = inference_data.to(device)

        logits = model(inference_data)

        x = torch.argmax(logits, dim=1).cpu().detach().numpy()
        result.append(x)
        # raise

    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({"filename": inference_set.filenames, "label": result})
    df.to_csv(os.path.join(outpath, filename), index=False)


if __name__ == "__main__":
    args = get_args()
    inference(args.input, args.output, args.model)
