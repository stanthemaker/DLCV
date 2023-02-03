import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
import argparse

from net import DANN
from dataset import USPSdataset, SVHNdataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument(
        "--expname", "-e", type=str, default="test", help="Experiment name"
    )
    parser.add_argument("--source", "-s", type=str, default="", help="source dataset")
    parser.add_argument("--target", "-t", type=str, default="", help="target dataset")
    parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")
    return parser.parse_args()


def test(datapath, valid_name, epoch, ckptpath, exp_name, device):
    target_data_path = os.path.join(datapath, valid_name)

    device = device if torch.cuda.is_available() else "cpu"
    batch_size = 128

    alpha = 0

    # valid set -> without labels
    if valid_name == "svhn":
        target_valid_set = SVHNdataset(target_data_path, train=False, source=False)
    else:
        target_valid_set = USPSdataset(target_data_path, train=False, source=False)

    target_valid_loader = DataLoader(
        target_valid_set, batch_size=batch_size, shuffle=True
    )

    model = torch.load(ckptpath)
    model = model.eval()
    loss_class = torch.nn.NLLLoss()

    n_total = 0
    n_correct = 0
    valid_loss = []
    for valid_data in tqdm(target_valid_loader):

        valid_image, valid_label = valid_data
        valid_image = valid_image.to(device)

        class_output, _ = model(input_data=valid_image, alpha=alpha)
        loss = loss_class(class_output, valid_label.to(device))
        pred = class_output.data.max(1, keepdim=True)[1]

        pred = pred.cpu()
        valid_label = valid_label.cpu()
        n_correct += pred.eq(valid_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        valid_loss.append(loss.item())

    accu = n_correct.data.numpy() * 1.0 / n_total

    print("epoch: %d, accuracy of the %s dataset: %f" % (epoch, valid_name, accu))
    with open(f"./{exp_name}_log.txt", "a") as f:
        f.write(
            f"epoch: {epoch + 1:03d}, accuracy of the {valid_name} dataset: {accu:.5f}\n"
        )
    # test(source_dataset_name, epoch)
    # test(target_dataset_name, epoch)
    return sum(valid_loss) / len(valid_loss), accu


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.01 / (1.0 + 10 * p) ** 0.75

    return optimizer


def main(exp_name: str, source_name: str, target_name: str, cuda: str):
    device = cuda if torch.cuda.is_available() else "cpu"
    params = {
        "batch_size": 256,  # Batch size during training.
        "nepochs": 500,  # Number of training epochs.
        "lr": 1e-4,  # Learning rate for optimizers
        "savepath": "/home/stan/hw2-stanthemaker/problem3_DANN/ckpt",  # save .pth every n epochs
        "data_dir": "/home/stan/hw2-stanthemaker/hw2_data/digits",
    }

    myseed = 1314520  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    # source set
    source_data_path = os.path.join(params["data_dir"], source_name)
    target_data_path = os.path.join(params["data_dir"], target_name)
    source_train_set = SVHNdataset(
        source_data_path, train=True, source=True
    )  # source training -> image and labels in train csv
    # target set -> without labels
    if target_name == "svhn":
        target_train_set = SVHNdataset(
            target_data_path, train=True, source=False
        )  # target training -> image in train csv
    else:
        target_train_set = USPSdataset(
            target_data_path, train=True, source=False
        )  # target training -> image in train csv

    source_train_loader = DataLoader(
        source_train_set, batch_size=params["batch_size"], shuffle=True, num_workers=1
    )
    target_train_loader = DataLoader(
        target_train_set, batch_size=params["batch_size"], shuffle=True, num_workers=1
    )

    model = DANN().to(device)
    # model = torch.load("./ckpt/usps_2_model_75.ckpt")

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, mode='max', patience=3, factor=0.5, min_lr=5e-6)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    best_source = 0
    best_target = 0
    n_epoch = params["nepochs"]
    for epoch in range(n_epoch):
        model.train()

        len_dataloader = min(len(source_train_loader), len(target_train_loader))
        data_source_iter = iter(source_train_loader)
        data_target_iter = iter(target_train_loader)

        for i in tqdm(range(len_dataloader), position=0, leave=True):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            optimizer = optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            # source data
            source_data = data_source_iter.next()
            source_image, source_label = source_data
            source_image = source_image.to(device)
            source_label = source_label.to(device)

            domain_label = torch.zeros(params["batch_size"])  # label for source domain
            domain_label = domain_label.long().to(device)

            class_output, domain_output = model(input_data=source_image, alpha=alpha)

            err_s_label = loss_class(class_output, source_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # target data
            target_data = data_target_iter.next()
            (
                target_image,
                _,
            ) = target_data
            target_image = target_image.to(device)

            domain_label = torch.ones(len(target_image))  # label for target domain
            domain_label = domain_label.long().to(device)

            _, domain_output = model(input_data=target_image, alpha=alpha)

            err_t_domain = loss_domain(domain_output, domain_label)

            # add all err
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()

            optimizer.step()

            print(
                "epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f"
                % (
                    epoch,
                    i,
                    len_dataloader,
                    err_s_label.cpu().data.numpy(),
                    err_s_domain.cpu().data.numpy(),
                    err_t_domain.cpu().data.numpy(),
                )
            )
        with open(f"./{exp_name}_log.txt", "a") as f:
            f.write(
                f"[ Train {exp_name} | {epoch + 1:03d}/{n_epoch:03d} ] err:{err:.5f}\n"
            )
        torch.save(
            model, "{0}/{1}_model_temp.ckpt".format(params["savepath"], exp_name)
        )
        valid_source_loss, acc_source = test(
            params["data_dir"],
            source_name,
            epoch,
            "{0}/{1}_model_temp.ckpt".format(params["savepath"], exp_name),
            exp_name,
            device,
        )
        valid_target_loss, acc_target = test(
            params["data_dir"],
            target_name,
            epoch,
            "{0}/{1}_model_temp.ckpt".format(params["savepath"], exp_name),
            exp_name,
            device,
        )

        if acc_target > best_target:
            best_target = acc_target
            torch.save(
                model,
                "{0}/{1}_model_{2}.ckpt".format(params["savepath"], exp_name, epoch),
            )
        elif acc_source > best_source:
            best_source = acc_source
            torch.save(
                model,
                "{0}/{1}_model_{2}.ckpt".format(params["savepath"], exp_name, epoch),
            )


if __name__ == "__main__":
    args = get_args()
    main(args.expname, args.source, args.target, args.cuda)

# train("../hw2_data/digits", "mnistm", "svhn", "./ckpt", "svhn_final") # adapt to svhn
# train("../hw2_data/digits", "mnistm", "usps", "./ckpt", "usps_final") # adapt to usps
