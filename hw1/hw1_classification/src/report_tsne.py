from net import CNN
import torch
from Dataset import VehilcleDataset
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
import numpy as np
import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import matplotlib.pyplot as plt
from sklearn import manifold


def generate_colormap(number_of_distinct_colors: int = 50):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(
        math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades
    )

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = (
        np.arange(number_of_distinct_colors_with_multiply_of_shades)
        / number_of_distinct_colors_with_multiply_of_shades
    )

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(
        number_of_shades,
        number_of_distinct_colors_with_multiply_of_shades // number_of_shades,
    )

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8 / lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = (
                np.ones(number_of_shades)
                - initial_cm[
                    lower_half
                    + j * number_of_shades : lower_half
                    + (j + 1) * number_of_shades,
                    i,
                ]
            )
            modifier = j * modifier / upper_partitions_half
            initial_cm[
                lower_half
                + j * number_of_shades : lower_half
                + (j + 1) * number_of_shades,
                i,
            ] += modifier

    return ListedColormap(initial_cm)


# Hints:
# Set features_extractor to eval mode
# Start evaluation and collect features and labels

activation = {}


def get_activation(name: str):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


device = torch.device("cuda:0")

model = CNN().to(device)
model.eval()
ckpt_path = "../ckpt/1009_10:52--CNN.ckpt"
name = "init"
model.load_state_dict(torch.load(ckpt_path))

data_dir = "../p1_data"

model.cnn[-1].register_forward_hook(get_activation("secondlast"))
valid_set = VehilcleDataset(os.path.join(data_dir, "val_50"), mode="valid")
valid_loader = DataLoader(
    valid_set, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
)
labels = []
features = torch.zeros((0, 8192), dtype=torch.float32)
features = []
for source_data, source_label in tqdm(valid_loader):
    source_data = source_data.to(device)
    feature = model(source_data)

    # labels.append(source_label)
    labels.extend(source_label.tolist())

    # if want to get output of second last
    feature = activation["secondlast"]
    # print(feature.shape)
    feature = feature.view(-1, 8192).cpu().detach().numpy()
    features.append(feature)
    # features = torch.cat((features, feature.detach().cpu()), dim=0)


features = np.concatenate(features)
labels = np.array(labels)
# print(features.shape)

"""## Step2: Apply t-SNE and normalize"""

# process extracted features with t-SNE
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
cmap = generate_colormap()
fig, ax = plt.subplots(figsize=(8, 8))
num_categories = 50
for lab in range(num_categories):
    indices = labels == lab
    ax.scatter(
        X_norm[indices, 0],
        X_norm[indices, 1],
        c=np.array(cmap(lab)).reshape(1, 4),
        label=lab,
        alpha=0.5,
    )
ax.legend(fontsize="large", markerscale=2)
plt.savefig(f"./tsne_{name}.png")
