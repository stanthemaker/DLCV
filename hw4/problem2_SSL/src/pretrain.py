import torch
from torchvision import models
from BYOL import BYOL
from dataset import ImgDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import argparse
import json
import random


def config_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ckpt", default="", help="ckpt to load")
    parser.add_argument("--name", default="test", help="pretrained model name")
    parser.add_argument("--cuda", default="cuda", help="choose a cuda")
    parser.add_argument(
        "--config",
        default="/home/stan/hw4-stanthemaker/problem2_SSL/src/config.json",
        help="path to config.json",
    )
    return parser


parser = config_parser()
args = parser.parse_args()
with open(args.config, "r") as f:
    config = json.load(f)

device = args.cuda if torch.cuda.is_available() else "cpu"
print(f"using device {device} {torch.cuda.current_device()}")
resnet = models.resnet50(weights=None).to(device)

if args.ckpt:
    state = torch.load(args.ckpt)
    # epoch = state["epoch"]
    model_state = state["model"]
    resnet.load_state_dict(model_state)
    # resnet = resnet.to(device)

# learner=BYOL(resnet,image_size=128,hidden_layer=“avgpool”)
learner = BYOL(resnet, image_size=128, hidden_layer="avgpool")
learner = learner
opt = torch.optim.Adam(learner.parameters(), lr=config["lr"])
train_set = ImgDataset(config["img_dir"])
train_loader = DataLoader(
    train_set,
    batch_size=config["batch_size"],
    pin_memory=True,
    shuffle=True,
    num_workers=0,
)
main_pb = tqdm(range(config["num_epochs"]))
name = args.name
epoch = 0
while epoch <= config["num_epochs"]:

    main_pb.set_description(f"epoch: {epoch:02d}")
    for images in tqdm(train_loader):
        images = images.to(device)
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()  # update moving average of target encoder

    with open(os.path.join(config["log_dir"], f"pretrain_{name}.txt"), "a") as f:
        f.write(
            f"[ Pretrain | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {loss:.4f}\n"
        )
    # save your improved network
    if epoch % 50 == 0:
        torch.save(
            resnet.state_dict(),
            os.path.join(config["ckpt_dir"], f"pretrain_{name}.ckpt"),
        )
    epoch += 1
