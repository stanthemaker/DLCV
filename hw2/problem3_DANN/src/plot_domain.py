import os

import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from dataset import SVHNdataset, USPSdataset
from net import DANN
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def test(datapath, valid_name, epoch, ckptpath, exp_name):
    target_data_path = os.path.join(datapath, valid_name)
    source_data_path = os.path.join(datapath, "mnistm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    alpha = 0

    source_valid_set = SVHNdataset(source_data_path, train=False, source=False)
    source_valid_loader = DataLoader(source_valid_set, batch_size=batch_size)
    # valid set -> without labels
    if valid_name == "svhn":
        target_valid_set = SVHNdataset(target_data_path, train=False, source=False)
    else:
        target_valid_set = USPSdataset(target_data_path, train=False, source=False)

    target_valid_loader = DataLoader(
        target_valid_set, batch_size=batch_size, shuffle=True
    )

    model = torch.load(ckptpath, map_location="cuda:0")
    model = model.eval()
    model.feature[-1].register_forward_hook(get_activation("feature_extractor"))
    loss_class = torch.nn.NLLLoss()

    n_total = 0
    n_correct = 0
    valid_loss = []
    target_features = torch.empty((0, 50, 4, 4), dtype=torch.float32).to(device)
    target_labels = torch.empty(0).to(device)
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

        # plot tsne
        feature = activation["feature_extractor"]
        target_features = torch.cat((target_features, feature), 0)
        target_labels = torch.cat((target_labels, valid_label.to(device)), 0)

    accu = n_correct.data.numpy() * 1.0 / n_total

    print("epoch: %d, accuracy of the %s dataset: %f" % (epoch, valid_name, accu))
    with open(f"./{exp_name}_log.txt", "a") as f:
        f.write(
            f"epoch: {epoch + 1:03d}, accuracy of the {valid_name} dataset: {accu:.5f}\n"
        )
    target_features = np.array(target_features.detach().cpu())
    target_features = target_features.reshape(len(target_features), 50 * 4 * 4)
    target_labels = np.array(target_labels.detach().cpu())

    # source
    source_features = torch.empty((0, 50, 4, 4), dtype=torch.float32).to(device)
    source_labels = torch.empty(0).to(device)
    for valid_data in tqdm(source_valid_loader):

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

        # plot tsne
        feature = activation["feature_extractor"]
        source_features = torch.cat((source_features, feature), 0)
        source_labels = torch.cat((source_labels, valid_label.to(device)), 0)

    accu = n_correct.data.numpy() * 1.0 / n_total

    print("epoch: %d, accuracy of the %s dataset: %f" % (epoch, valid_name, accu))
    with open(f"./{exp_name}_log.txt", "a") as f:
        f.write(
            f"epoch: {epoch + 1:03d}, accuracy of the {valid_name} dataset: {accu:.5f}\n"
        )
    source_features = np.array(source_features.detach().cpu())
    source_features = source_features.reshape(len(source_features), 50 * 4 * 4)
    source_labels = np.array(source_labels.detach().cpu())

    fig, ax = plt.subplots(figsize=(16, 16))

    # target -> red
    X_tsne = manifold.TSNE(
        n_components=2, init="random", random_state=5, verbose=1
    ).fit_transform(target_features)

    # Normalization the processed features
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    """## Step3: Visualization with matplotlib"""

    # Data Visualization
    # Use matplotlib to plot the distribution
    # The shape of X_norm is (N,2)
    # cmap = plt.cm.get_cmap("Spectral")
    num_categories = 10
    cm = plt.cm.get_cmap("Set1")
    ax.scatter(
        X_norm[:, 0],
        X_norm[:, 1],
        c=np.array(cm(0)).reshape(1, 4),
        label=0,
        alpha=0.5,
    )

    # source -> blue
    X_tsne = manifold.TSNE(
        n_components=2, init="random", random_state=5, verbose=1
    ).fit_transform(source_features)

    # Normalization the processed features
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    """## Step3: Visualization with matplotlib"""

    # Data Visualization
    # Use matplotlib to plot the distribution
    # The shape of X_norm is (N,2)
    # cmap = plt.cm.get_cmap("Spectral")
    num_categories = 10
    cm = plt.cm.get_cmap("Set1")
    ax.scatter(
        X_norm[:, 0],
        X_norm[:, 1],
        c=np.array(cm(1)).reshape(1, 4),
        label=1,
        alpha=0.5,
    )
    ax.legend(fontsize="large", markerscale=2)
    plt.savefig("./plot.png")

    return sum(valid_loss) / len(valid_loss), accu


if __name__ == "__main__":
    # test("../hw2_data/digits", "usps", 0,"./ckpt/usps_4_model_46.ckpt", 0) # 0.8177
    test(
        "/home/stan/hw2-stanthemaker/hw2_data/digits",
        "usps",
        0,
        "/home/stan/hw2-stanthemaker/problem3_DANN/ckpt/usps_f_model_best.ckpt",
        0,
    )  # 0.54368
