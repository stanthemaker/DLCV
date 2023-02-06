import csv
import pandas as pd
import argparse
def get_args():
    argparser = argparse.ArgumentParser(description="Evalai challenge: Talking to me")


    argparser.add_argument("--threshold", type=float, default=0.5)
    argparser.add_argument("--input_file", type=str, default="./output.csv")
    return argparser

def generateCSV(args):
    threshold = args.threshold
    inputfile = args.input_file
    outputfile = f"./pred_{threshold}.csv"
    pred = []
    with open(
        inputfile,
        mode="r",
    ) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:  # skip header
                line_count += 1
            pred.append([row["Id"], row["Predicted"]])
    for i in range(len(pred)):
        if float(pred[i][1]) >= threshold:
            pred[i][1] = 1
        else:
            pred[i][1] = 0
    pred = pd.DataFrame(pred)
    pred.columns = ["Id", "Predicted"]
    pred.to_csv(outputfile, index=False)



if __name__ == "__main__":
    args = get_args().parse_args()
    generateCSV(args)