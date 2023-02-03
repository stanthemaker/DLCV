# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import sys
import argparse

# This is for the progress bar.
from tqdm.auto import tqdm
from datetime import datetime

# self-defined modules
from Dataset import VehilcleDataset
from net import CNN, Inceptionv3


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument(
        "--expname", "-e", type=str, default="", help="Load test data for evaluation"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")
    return parser.parse_args()


def main(exp_name: str, model_path: str, cuda: str):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    myseed = 1314520  # set a random seed for reproducibility
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    device = torch.device(cuda)

    exp_name = exp_name

    model = Inceptionv3()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    now = datetime.now()
    dt_string = now.strftime("%m%d_%H:%M")
    config = {
        "model_type": "scratch",
        "Optimizer": "Adam",
        "batch_size": 64,
        "lr": 1e-6,
        "n_epochs": 100,
        "patience": 30,
        "exp_name": dt_string + exp_name,
    }

    _dataset_dir = "../p1_data"
    # Construct datasets.
    # The argument "loader" tells how torchvision reads the data.
    train_set = VehilcleDataset(os.path.join(_dataset_dir, "train_50"), mode="train")
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    valid_set = VehilcleDataset(os.path.join(_dataset_dir, "val_50"), mode="valid")
    valid_loader = DataLoader(
        valid_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    model = Inceptionv3()
    model = model.to(device)

    # checkpoint = torch.load("inception.pt")
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # init_epoch = checkpoint["epoch"]
    # loss = checkpoint["loss"]

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0.87
    n_epochs = config["n_epochs"]
    _exp_name = config["exp_name"]
    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # for inceptionv3

            logits = model(imgs.to(device))

            # logits = model(imgs.to(device))
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        with open(f"../ckpt/log/{_exp_name}_log.txt", "a") as f:
            f.write(
                f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}"
            )
            print(
                f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}"
            )

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()
            labels = labels.to(device)
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # get PCA

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels)

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            # break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        # update logs

        if valid_acc > best_acc:
            with open(f"../ckpt/log/{_exp_name}_log.txt", "a") as f:
                f.write(
                    f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}-> best \n"
                )
                print(
                    f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best"
                )
        else:
            with open(f"../ckpt/log/{_exp_name}_log.txt", "a") as f:
                f.write(
                    f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} \n"
                )
                print(
                    f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
                )

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(
                model.state_dict(), f"../ckpt/{_exp_name}.ckpt"
            )  # only save best to prevent output memory exceed error
            best_acc = valid_acc
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


if __name__ == "__main__":
    args = get_args()
    main(args.expname, args.model, args.cuda)
