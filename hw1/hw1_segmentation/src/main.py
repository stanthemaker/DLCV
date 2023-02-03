
# Basic packages.
import numpy as np
import pandas as pd
import torch
import os
from datetime import datetime
import sys

# Torch related packages
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision
from tqdm.auto import tqdm
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
import random
import imageio
from PIL import Image

# Customized packages.
from net import Deeplabv3, FCN32_VGG16
from Dataset import groundDataset
from utils import mean_iou_score

exp_name = "FCN32"
now = datetime.now()
dt_string = now.strftime("%m%d_%H:%M_")
config = {
    "Optimizer": "Adam",
    "batch_size": 8,
    "lr": 2e-5,
    "n_epochs": 100,
    "patience": 20,
    "transforms": "None",
    "exp_name": dt_string + exp_name,
}
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
myseed = 1314520  # set a random seed for reproducibility
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
device = "cuda" if torch.cuda.is_available() else "cpu"


def read_masks(file_list):
    n_masks = len(file_list)
    masks = torch.empty(
        (n_masks, 512, 512), dtype=torch.int)

    for i, file in enumerate(file_list):
        mask = imageio.imread(file)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3  # (Green: 010) Forest land
        masks[i, mask == 1] = 4  # (Blue: 001) Water
        masks[i, mask == 7] = 5  # (White: 111) Barren land
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
    # print(masks.shape)
    return masks

def transform(image: Image, mask: torch.Tensor ,aug:bool):
        mask  = TF.to_pil_image(mask)

        if random.random() > 0.5 and aug:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5 and aug:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, torch.squeeze(mask)

_dataset_dir = "../input/dlcvhw1/hw1_data/hw1_data/p2_data"
train_set = groundDataset(
    path=os.path.join(_dataset_dir,"train"), mode = "train"
)
valid_set = groundDataset(
    path=os.path.join(_dataset_dir, "validation" ), mode = "validation"
)
train_loader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"])
valid_loader = DataLoader(valid_set, shuffle=True, batch_size=config["batch_size"])
print(len(train_set))
print(len(valid_set))

def mean_iou_score(preds: np, labels: np) -> float:
    """
    Compute mean IoU score over 6 classes
    """

    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(preds == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((preds == i) * (labels == i))
        # print("numbers of tp, tp_fp, tp_fn", tp, tp_fp, tp_fn)
        iou = tp / (tp_fp + tp_fn - tp + 1e-6)
        mean_iou += iou / 6

    return mean_iou
# model = FCN32_VGG16()
model = Classifier(num_classes = 7)
model = model.to(device)

ckpt_path = "../input/dlcvhw1/1008_04_52_deeplabv3.ckpt"
model.load_state_dict(torch.load(ckpt_path))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)

stale = 0
best_acc = 0.73
n_epochs = config["n_epochs"]
_exp_name = config["exp_name"]
for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for i, batch in tqdm(enumerate(train_loader)):
        # A batch consists of image data and corresponding labels.
        images, true_masks = batch
        true_masks = true_masks.to(device)
        # print(true_masks.shape)
        
        images = images.to(device)
        # Forward the model.
        logits = model(images)
        loss = criterion(logits, true_masks.long())
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        loss = 0
        # Record information.
        output = logits.argmax(dim=len(logits.shape) - 3)
        pixel_wise_acc = (output == true_masks).float().mean()
        # train_loss.append(loss.item())
        train_loss.append(loss)
        train_accs.append(pixel_wise_acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
   
   
    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, pixel-wised_acc = {train_acc:.5f}"
    )
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    outputs, labels = np.empty((len(valid_set), 512, 512)), np.empty(
        (len(valid_set), 512, 512)
    )

    # Iterate the validation set by batches.
    for i, batch in tqdm(enumerate(valid_loader)):

        images, true_masks = batch
        true_masks = true_masks.to(device, dtype=torch.long)
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        
        loss = criterion(logits, true_masks.long())
        output = logits.argmax(dim=len(logits.shape) - 3)
        pixel_wise_acc = (output == true_masks).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(pixel_wise_acc)
        outputs[
            i * config["batch_size"] : i * config["batch_size"] + output.shape[0],
            :,
            :,
        ] = (
            output.cpu().detach().numpy()
        )

        labels[
            i * config["batch_size"] : i * config["batch_size"] + output.shape[0],
            :,
            :,
        ] = (
            true_masks.cpu().detach().numpy()
        )

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    miou_acc = sum(valid_accs) / len(valid_accs)
    miou_acc = mean_iou_score(outputs, labels)

    # update logs
    if miou_acc > best_acc:
      print(
          f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, pixel-wised_acc = {pixel_wise_acc:.5f}, miou_acc = {miou_acc:5f} -> best"
      )
    else:
      print(
          f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, pixel-wised_acc = {pixel_wise_acc:.5f}, miou_acc = {miou_acc:5f}"
      )

    # save models
    if miou_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"./{_exp_name}.ckpt") 
        best_acc = miou_acc
        stale = 0

    else:
        stale += 1
        patience = config["patience"]
        if stale > patience:
            print(
                f"No improvment {patience} consecutive epochs, early stop in {epoch} epochs"
            )
            break

print(f"Training done with best accuracy: {best_acc}")
print("Training details:")
print(config)

