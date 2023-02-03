import torch
from dataset import MNISTdataset

config = {
    "data_dir": "../input/hw2-data/hw2_data/hw2_data/digits/mnistm/data",
    "dfname": "../input/hw2-data/hw2_data/hw2_data/digits/mnistm/train.csv",
    "lr": 1e-4,
    "num_epochs": 100,
    "batch_size": 256,
    "num_classes": 10,
    "n_T": 300,
    "img_size": 28,
    "output_dir": "./outputs",
    "sample_dir": "./samples",
    "ckpt_dir": "./",
    #     'classifier_path':"../input/Classifier.pth"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")
dataset = MNISTdataset(config["data_dir"], config["dfname"], config["img_size"])
