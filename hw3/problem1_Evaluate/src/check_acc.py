import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Evaluate predicted csv")
parser.add_argument("--path", "-p", type=str, default=None, help="csv path")
args = parser.parse_args()

df = pd.read_csv(args.path)
correct = 0
for i, row in df.iterrows():
    if int(row["filename"].split("_")[0]) == row["label"]:
        correct += 1

print(correct / len(df))
