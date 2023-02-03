import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from model import Transformer

# from new_model import build_model
from trainer import Trainer, seed_everything
from dataset import (
    ImgCaptrionDataset,
    collate_padd,
)
from tokenizers import Tokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tok",
    "-t",
    type=str,
    default="/home/stan/hw3-stanthemaker/hw3_data/caption_tokenizer.json",
    help="tokenizer path to load",
)
parser.add_argument("--model", "-m", type=str, default="", help="Model path to load")
parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")
parser.add_argument("--name", "-n", type=str, default="test", help="train name")
config = {
    "seed": 1314520,
    "max_len": 55,
    "batch_size": 32,
    "num_epochs": 100,
    "val_interval": 2,
    "pad_id": 0,
    "vocab_size": 18202,
    "d_model": 768,  # 768
    "dec_ff_dim": 2048,
    "dec_n_layers": 6,
    "dec_n_heads": 12,  # 12 # 8
    "dropout": 0.1,
    "max_norm": 0.1,
    "save_path": "/home/stan/hw3-stanthemaker/problem2_ImageCaptioning/ckpt",
}

args = parser.parse_args()
device = torch.device(args.cuda if (torch.cuda.is_available()) else "cpu")
print(f"using device {device}")
tokenizer_file = args.tok
tokenizer = Tokenizer.from_file(tokenizer_file)

print("start to create datasets")
train_set = ImgCaptrionDataset(
    "/home/stan/hw3-stanthemaker/hw3_data/p2_data/images/train",
    tokenizer,
    "/home/stan/hw3-stanthemaker/hw3_data/p2_data/train.json",
    384,
)
valid_set = ImgCaptrionDataset(
    "/home/stan/hw3-stanthemaker/hw3_data/p2_data/images/val",
    tokenizer,
    "/home/stan/hw3-stanthemaker/hw3_data/p2_data/val.json",
    384,
    False,
)


train_loader = DataLoader(
    train_set,
    collate_fn=collate_padd(config["max_len"], config["pad_id"]),
    batch_size=config["batch_size"],
    pin_memory=True,
    shuffle=True,
    num_workers=0,
)
valid_loader = DataLoader(
    valid_set,
    collate_fn=collate_padd(config["max_len"], config["pad_id"]),
    batch_size=1,
    pin_memory=True,
    shuffle=True,
    num_workers=0,
)
print("finished creating datasets")

print("start to create model")
transformer = Transformer(
    config["vocab_size"],
    config["d_model"],
    config["dec_ff_dim"],
    config["dec_n_layers"],
    config["dec_n_heads"],
    config["max_len"] - 1,
)
print("finished creating model")


SEED = config["seed"]
seed_everything(SEED)

trainer = Trainer(
    transformer,
    device,
    config["num_epochs"],
    config["val_interval"],
    config["max_norm"],
    config["save_path"],
    config["pad_id"],
    args.model,
)
print("start training")
trainer.run_train(
    args.name,
    transformer,
    tokenizer,
    (train_loader, valid_loader),
    SEED,
)
