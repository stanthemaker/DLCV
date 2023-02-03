# https://www.kaggle.com/code/b09901104/dlcvhw2-problem1/edit
# Import necessary  standard packages.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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
from net import WGANGP_Discriminator, WGANGP_Generator , compute_gp


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
        "--save", "-s", type=int, default=1, help="determine if want to save model"
    )
    return parser.parse_args()


def main(exp_name: str, model_path: str, cuda: str, to_save: bool):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # Parameters to define the model.
    time = datetime.now().strftime("%m%d-%H%M_")
    train_name = time + exp_name
    params = {
        "batch_size": 128,  # Batch size during training.
        "imsize": 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
        "nc": 3,  # Number of channles in the training images. For coloured images this is 3.
        "nz": 100,  # Size of the Z latent vector (the input to the generator).
        "nepochs": 100,  # Number of training epochs.
        "lr": 0.0002,  # Learning rate for optimizers
        "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
        "n_critic": 2,  # The number of training steps for discriminator per iter
        "sample_interval": 2,  # save smaple outputs every n epochs
        "lambda_gp": 2,
    }  # Save step.

    # Use GPU is available else use CPU.
    device = torch.device(cuda if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Get the data.
    data_dir = "/home/stan/hw2-stanthemaker/hw2_data/face/"
    dataset = PhotoDataset(path=os.path.join(data_dir, "train"), mode="train")
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    if model_path:
        state_dict = torch.load(model_path)
        params = state_dict["params"]
        generator = WGANGP_Generator(params["nz"]).to(device)
        discriminator = WGANGP_Discriminator(params["nc"]).to(device)
        optimizer_D = torch.optim.Adam(discriminator.parameters())
        optimizer_G = torch.optim.Adam(generator.parameters())
        optimizer_D.load_state_dict(state_dict["optimizerD"])
        optimizer_G.load_state_dict(state_dict["optimizerG"])

    else:
        generator = WGANGP_Generator(params["nz"]).to(device)
        discriminator = WGANGP_Discriminator(params["nc"]).to(device)
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
        )
        optimizer_G = torch.optim.Adam(
            generator.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
        )
    print(generator)
    print(discriminator)
    # Binary Cross Entropy loss function.
    G_losses = []
    D_losses = []
    n_epochs = params["nepochs"]
    Tensor = torch.FloatTensor
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
            z = Variable(torch.randn(bs, params["nz"])).to(device)
            r_imgs = Variable(data.type(Tensor)).to(device)
            f_imgs = generator(z)

            # Model forwarding
            r_logit = discriminator(r_imgs.detach())
            f_logit = discriminator(f_imgs.detach())

            gradient_penalty = compute_gp(
                discriminator, r_imgs.data, f_imgs.data, device
            )
            loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
            # print("r logits",torch.mean(r_logit).item())
            # print("f logits",torch.mean(f_logit).item())
            # print("gp :",gradient_penalty.item())
            # print("loss_D :",loss_D.item())

            discriminator.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            D_losses.append(loss_D.item())

            if i % params["n_critic"] == 0:
                # ==========================
                #  Train G
                # ==========================
                f_imgs = generator(z)
                f_logit = discriminator(f_imgs)
                loss_G = -torch.mean(f_logit)
                loss_G.backward()
                generator.zero_grad()
                optimizer_G.step()
                G_losses.append(loss_G.item())

        D_loss = sum(D_losses) / len(D_losses)
        G_loss = sum(G_losses) / len(G_losses)
        # ------------------- an epoch finish ---------------------#
        with open(
            f"/home/stan/hw2-stanthemaker/problem1_GAN/log/{train_name}_log.txt", "a"
        ) as f:
            f.write(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] D_loss = {D_loss:.4f}, G_loss = {G_loss:.4f}\n"
            )
            print(
                f"[ {epoch + 1:03d}/{n_epochs:03d} ] D_loss = {D_loss:.4f}, G_loss = {G_loss:.4f}"
            )

        if epoch % params["sample_interval"] == 0:
            generator.eval()
            z = Variable(Tensor(np.random.normal(0, 1, (32, params["nz"])))).to(device)
            f_imgs = generator(z)
            sample_path = "/home/stan/hw2-stanthemaker/problem1_GAN/train_output/"
            filename = os.path.join(sample_path, f"wgangp_report32_{epoch}.png")
            save_image(
                f_imgs.data,
                filename,
                nrow=8,
                normalize=True,
            )
            generator.train()
    # *********************
    # *    inference        *
    # *********************
    # n_outputs = 1000
    # output_path = "/home/stan/hw2-stanthemaker/problem1_GAN/inference_output"
    # z = Variable(Tensor(np.random.normal(0, 1, (n_outputs, params["nz"])))).to(device)
    # sample_z = z[:32]

    # unorm = UnNormalize()
    # generator.eval()
    # with torch.no_grad():
    #     # *********************
    #     # *    report      *
    #     # *********************
    #     f_imgs_sample = ((generator(sample_z)).data + 1) / 2.0
    #     filename = os.path.join(
    #         "/home/stan/hw2-stanthemaker/problem1_GAN/sample_output",
    #         f"{train_name}_32.png",
    #     )
    #     vutils.save_image(f_imgs_sample, filename, nrow=8)

    #     data = (generator(z)).detach().cpu()
    #     for i in range(n_outputs):
    #         vutils.save_image(unorm(data[i]), os.path.join(output_path, f"{i}.png"))


if __name__ == "__main__":
    args = get_args()
    main(args.expname, args.model, args.cuda, args.save)
