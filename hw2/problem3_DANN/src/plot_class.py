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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    alpha = 0

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
    features = torch.empty((0, 50, 4, 4), dtype=torch.float32).to(device)
    labels = torch.empty(0).to(device)
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
        features = torch.cat((features, feature), 0)
        labels = torch.cat((labels, valid_label.to(device)), 0)

    accu = n_correct.data.numpy() * 1.0 / n_total

    print("epoch: %d, accuracy of the %s dataset: %f" % (epoch, valid_name, accu))
    with open(f"./{exp_name}_log.txt", "a") as f:
        f.write(
            f"epoch: {epoch + 1:03d}, accuracy of the {valid_name} dataset: {accu:.5f}\n"
        )
    # test(source_dataset_name, epoch)
    # test(target_dataset_name, epoch)

    features = np.array(features.detach().cpu())
    features = features.reshape(len(features), 50 * 4 * 4)
    labels = np.array(labels.detach().cpu())
    X_tsne = manifold.TSNE(
        n_components=2, init="random", random_state=5, verbose=1
    ).fit_transform(features)

    # Normalization the processed features
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    """## Step3: Visualization with matplotlib"""

    # Data Visualization
    # Use matplotlib to plot the distribution
    # The shape of X_norm is (N,2)
    # cmap = plt.cm.get_cmap("Spectral")
    fig, ax = plt.subplots(figsize=(16, 16))
    num_categories = 10
    cm = plt.cm.get_cmap("tab10")
    for lab in range(num_categories):
        indices = labels == lab
        # print(indices)
        ax.scatter(
            X_norm[indices, 0],
            X_norm[indices, 1],
            c=np.array(cm(lab)).reshape(1, 4),
            label=lab,
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
