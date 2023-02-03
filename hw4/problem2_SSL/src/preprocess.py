import pandas as pd
import json

df = pd.read_csv('/home/stan/hw4-stanthemaker/hw4_data/office/train.csv')
labels = df['label'].unique()
labels = sorted(labels)
print(labels)
label_ids = {}

for i, label in enumerate(labels):
    label_ids[label] = i

with open("/home/stan/hw4-stanthemaker/hw4_data/office/label_ids.json", "w") as jsonfile:
    json.dump(label_ids, jsonfile)
