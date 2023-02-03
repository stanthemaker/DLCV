import os
from test import test

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from dataset import SVHNdataset, USPSdataset
from net import DANN


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


def trainOnTarget(datapath, source_name, target_name, savepath, exp_name):
    source_data_path = os.path.join(datapath, source_name)
    target_data_path = os.path.join(datapath, target_name)
    n_epoch = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    lr = 1e-4

    myseed = 1314520  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    # target set -> without labels
    if target_name == "svhn":
        target_train_set = SVHNdataset(
            target_data_path, train=True, source=False
        )  # target training -> image in train csv
    else:
        target_train_set = USPSdataset(
            target_data_path, train=True, source=False
        )  # target training -> image in train csv

    target_train_loader = DataLoader(
        target_train_set, batch_size=batch_size, shuffle=True, num_workers=1
    )

    model = DANN().to(device)
    # model = torch.load("./ckpt/usps_2_model_75.ckpt")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, mode='max', patience=3, factor=0.5, min_lr=5e-6)

    loss_class = torch.nn.NLLLoss()
    best_target = 0
    for epoch in range(n_epoch):
        model.train()

        len_dataloader = len(target_train_loader)
        data_target_iter = iter(target_train_loader)

        for i in tqdm(range(len_dataloader), position=0, leave=True):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            optimizer = optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            # target data
            target_data = data_target_iter.next()
            (
                target_image,
                target_label,
            ) = target_data
            target_image = target_image.to(device)
            target_label = target_label.to(device)

            class_output, _ = model(input_data=target_image, alpha=alpha)

            err_t_label = loss_class(class_output, target_label)

            # add all err
            err = err_t_label
            err.backward()

            optimizer.step()

        with open(f"./log/{exp_name}_log.txt", "a") as f:
            f.write(
                f"[ Train {exp_name} | {epoch + 1:03d}/{n_epoch:03d} ] err:{err:.5f}\n"
            )
        torch.save(model, "{0}/{1}_model_temp.ckpt".format(savepath, exp_name))

        _, acc_target = test(
            datapath,
            target_name,
            epoch,
            "{0}/{1}_model_temp.ckpt".format(savepath, exp_name),
            exp_name,
        )

        if acc_target > best_target:
            best_target = acc_target
            torch.save(model, "{0}/{1}_model.ckpt".format(savepath, exp_name))
