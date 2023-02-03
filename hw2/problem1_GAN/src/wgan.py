# Import necessary  standard packages.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import random
import os
import torchvision.utils as vutils
from torch.autograd import Variable


# This is for the progress bar.
from tqdm.auto import tqdm
from datetime import datetime

# self-defined modules
from Dataset import PhotoDataset, UnNormalize
from net import WGAN_Discriminator, WGAN_Generator
from utils.compute_gp import compute_gradient_penalty

# from utils.ImageScore import calculate_fid_given_paths, face_recog


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument(
        "--expname", "-e", type=str, default="test", help="Experiment name"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")
    parser.add_argument(
        "--save", "-s", type=int, default=1, help="if user wants to save model"
    )
    return parser.parse_args()


def main(exp_name: str, model_path: str, cuda: str, to_save: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # seed = 1
    seed = 1314520
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # Parameters to define the model.
    time = datetime.now().strftime("%m%d-%H%M_")
    train_name = time + exp_name
    params = {
        "batch_size": 128,  # Batch size during training.
        "imsize": 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
        "nc": 3,  # channles in the training images
        "nz": 100,  # Size of the Z latent vector (the input to the generator).
        "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
        "ndf": 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
        "n_critic": 2,
        "nepochs": 100,  # Number of training epochs.
        "lr": 1e-4,  # Learning rate for optimizers
        "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
        "sample_interval": 2,  # sample every n epochs
        "lambda": 10,
        "clip_value": 1,
        "output_path": "/home/stan/hw2-stanthemaker/problem1_GAN/train_output",  # save .pth every n epochs
        "data_dir": "/home/stan/hw2-stanthemaker/hw2_data/face/",
    }
    device = torch.device(cuda if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    dataset = PhotoDataset(path=os.path.join(params["data_dir"], "train"), mode="train")
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    if model_path:
        state_dict = torch.load(model_path)
        pre_params = state_dict["params"]
        generator = WGAN_Generator(in_dim=pre_params["nz"]).to(device)
        discriminator = WGAN_Discriminator(in_dim=pre_params["nc"]).to(device)
        generator.load_state_dict(state_dict["generator"])
        discriminator.load_state_dict(state_dict["discriminator"])
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=params["lr"])
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=params["lr"])
        optimizer_D.load_state_dict(state_dict["optimizerD"])
        optimizer_G.load_state_dict(state_dict["optimizerG"])

    else:
        generator = WGAN_Generator(in_dim=params["nz"]).to(device)
        discriminator = WGAN_Discriminator(in_dim=params["nc"]).to(device)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=params["lr"])
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=params["lr"])
    print(generator)
    print(discriminator)

    # Binary Cross Entropy loss function.
    # criterion = nn.BCELoss()
    # To denormalize inferenced images
    unorm = UnNormalize()

    G_losses = []
    D_losses = []
    n_epochs = params["nepochs"]

    print("Starting Training Loop...")
    print("-" * 25)
    for epoch in range(n_epochs):
        progress_bar = tqdm(dataloader)
        progress_bar.set_description(f"Epoch {epoch+1}")
        for i, data in enumerate(progress_bar):
            imgs = data
            imgs = imgs.to(device)
            bs = imgs.size(0)

            # ============================================
            #  Train D
            # ============================================
            discriminator.zero_grad()
            z = Variable(torch.randn(bs, params["nz"])).to(device)
            r_imgs = Variable(imgs).to(device)
            f_imgs = generator(z)

            r_logit = discriminator(r_imgs.detach())
            f_logit = discriminator(f_imgs.detach())
            gp = compute_gradient_penalty(
                discriminator, r_imgs.data, f_imgs.data, device
            )
            loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + params["lambda"] * gp
            loss_D.backward()
            optimizer_D.step()

            D_losses.append(loss_D.item())

            for p in discriminator.parameters():
                p.data.clamp_(-params["clip_value"], params["clip_value"])
            # ============================================
            #  Train G
            # ============================================
            if (epoch & params["n_critic"]) == 0:
                generator.zero_grad()

                z = Variable(torch.randn(bs, params["nz"])).to(device)
                f_imgs = generator(z)
                f_logit = discriminator(f_imgs)
                loss_G = -torch.mean(f_logit)
                loss_G.backward()
                optimizer_G.step()

                G_losses.append(loss_G.item())

                # Model forwarding

            # Save the losses for printing.

        # ------------------- an epoch finish ---------------------#
        G_loss = sum(G_losses) / len(G_losses)
        D_loss = sum(D_losses) / len(D_losses)

        # ------------------- generate 1000 images and evaluate---------------------#
        if epoch % params["sample_interval"] == 0:
            z = Variable(torch.randn(32, params["nz"])).to(device)
            f_imgs = generator(z).mul(0.5).add(0.5).detach().cpu()
            filename = os.path.join(
                "/home/stan/hw2-stanthemaker/problem1_GAN/",
                "report_32.png",
            )
            vutils.save_image(f_imgs, filename, nrow=8)

        with open(
            f"/home/stan/hw2-stanthemaker/problem1_GAN/log/{train_name}_log.txt", "a"
        ) as f:
            f.write(
                f"[ {epoch + 1:03d}/{n_epochs:03d} D_loss:{D_loss} , G_loss:{G_loss}] \n"
            )
            print(
                f"[ {epoch + 1:03d}/{n_epochs:03d} D_loss:{D_loss} , G_loss:{G_loss}]"
            )


if __name__ == "__main__":
    args = get_args()
    main(args.expname, args.model, args.cuda, args.save)
