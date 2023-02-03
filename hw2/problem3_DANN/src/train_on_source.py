import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from dataset import USPSdataset, SVHNdataset
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
    with open(f"./log/{exp_name}_log.txt", "a") as f:
        f.write(
            f"epoch: {epoch + 1:03d}, accuracy of the {valid_name} dataset: {accu:.5f}\n"
        )
    # test(source_dataset_name, epoch)
    # test(target_dataset_name, epoch)
    return sum(valid_loss) / len(valid_loss), accu


def trainOnSource(datapath, source_name, target_name, savepath, exp_name):
    source_data_path = os.path.join(datapath, source_name)
    target_data_path = os.path.join(datapath, target_name)
    n_epoch = 50

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    batch_size = 256
    lr = 1e-4

    myseed = 1314520  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    # source set
    source_train_set = USPSdataset(
        source_data_path, train=True, source=True
    )  # source training -> image and labels in train csv

    source_train_loader = DataLoader(
        source_train_set, batch_size=batch_size, shuffle=True, num_workers=1
    )
    model = DANN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    best_target = 0
    for epoch in range(n_epoch):
        model.train()

        len_dataloader = len(source_train_loader)
        data_source_iter = iter(source_train_loader)

        for i in tqdm(range(len_dataloader), position=0, leave=True):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            optimizer = optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            # source data
            source_data = data_source_iter.next()
            source_image, source_label = source_data
            source_image, source_label = source_image.to(device), source_label.to(
                device
            )

            class_output, _ = model(input_data=source_image, alpha=alpha)

            err_s_label = loss_class(class_output, source_label)

            err = err_s_label
            err.backward()

            optimizer.step()

            # print(
            #     "epoch: %d, [iter: %d / all %d], err_s_label: %f"
            #     % (
            #         epoch,
            #         i,
            #         len_dataloader,
            #         err_s_label.cpu().data.numpy(),
            #     )
            # )
        with open(f"./log/{exp_name}_log.txt", "a") as f:
            f.write(
                f"[ Train {exp_name} | {epoch + 1:03d}/{n_epoch:03d} ] err:{err:.5f}\n"
            )
        torch.save(model, "{0}/{1}_tmp.ckpt".format(savepath, exp_name))

        _, acc_target = test(
            datapath,
            target_name,
            epoch,
            "{0}/{1}_tmp.ckpt".format(savepath, exp_name),
            exp_name,
            device,
        )

        if acc_target > best_target:
            best_target = acc_target
            torch.save(model, "{0}/{1}_model.ckpt".format(savepath, exp_name))


trainOnSource(
    "/home/stan/hw2-stanthemaker/hw2_data/digits",
    "mnistm",
    "usps",
    "/home/stan/hw2-stanthemaker/problem3_DANN/ckpt",
    "",
)
