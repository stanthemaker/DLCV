import pandas as pd
import argparse


def config_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", default="", help="to check acc")
    return parser


parser = config_parser()
args = parser.parse_args()
df = pd.read_csv(args.csv)
correct = 0
for i, row in df.iterrows():
    if row["filename"][:-9] == row["label"]:
        correct += 1
print(correct / len(df))
