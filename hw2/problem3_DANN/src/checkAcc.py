import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument(
        "--pred", "-p", type=str, default="", help="path to prediction csv"
    )
    parser.add_argument(
        "--valid", "-v", type=str, default="", help="path to prediciton csv"
    )
    return parser.parse_args()


def evaluate(pred_path, valid_path):

    val = pd.read_csv(valid_path)
    val_label = val.sort_values("image_name").label.values[:]
    val_files = val.sort_values("image_name").image_name.values[:]

    test = pd.read_csv(pred_path)
    test_label = test.sort_values("image_name").label.values[:]
    test_files = test.sort_values("image_name").image_name.values[:]
    print(test_label[0])
    n_correct = 0
    n = len(val)

    test = test.reset_index()
    for idx, row in test.iterrows():
        if row["image_name"] in val_files:
            # print(row["image_name"], val.loc[val["image_name"] == row["image_name"]].label.values[0], row["label"])
            if (
                val.loc[val["image_name"] == row["image_name"]].label.values[0]
                == row["label"]
            ):
                n_correct += 1
    print(n_correct, n, n_correct / n)


if __name__ == "__main__":
    args = get_args()
    evaluate(args.pred, args.valid)
