import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained deeplabv3_resetnet101 model."
    )
    parser.add_argument("--path", "-t", type=str, default=None, help="csv path")


def get_acc(path: str):
    df = pd.read_csv(path)
    correct = 0
    for i, row in df.iterrows():
        if int(row["filename"].split("_")[0]) == row["label"]:
            correct += 1

    return correct / len(df)


df = pd.read_csv("./pred.csv")
correct = 0
for i, row in df.iterrows():
    if int(row["filename"].split("_")[0]) == row["label"]:
        correct += 1

print(correct / len(df))
