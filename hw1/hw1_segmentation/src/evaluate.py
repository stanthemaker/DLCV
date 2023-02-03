import os, argparse
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader, Dataset
from PIL import Image

from net import Deeplabv3
from torchvision.transforms import functional as TF
from torchvision.utils import save_image


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained Deeplabv3 model."
    )
    parser.add_argument(
        "--input", "-t", type=str, default=None, help="Load test data for evaluation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory path for predition masks",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    return parser.parse_args()


def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def from_one_hot_to_rgb(one_hot_tensor: torch.Tensor) -> None:
    """Assign a different color to each class in the input tensor"""
    # masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
    # masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
    # masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
    # masks[i, mask == 2] = 3  # (Green: 010) Forest land
    # masks[i, mask == 1] = 4  # (Blue: 001) Water
    # masks[i, mask == 7] = 5  # (White: 111) Barren land
    # masks[i, mask == 0] = 6  # (Black: 000) Unknown
    H, W = one_hot_tensor.shape
    # cmap = one_hot_tensor.to(float)[None, :].expand(3, -1, -1)
    cmap = torch.empty(H, W, 3).to(float)
    # cmap : 3 x H x W
    cmap[one_hot_tensor == 0] = torch.tensor([0, 1, 1]).to(float)
    cmap[one_hot_tensor == 1] = torch.tensor([1, 1, 0]).to(float)
    cmap[one_hot_tensor == 2] = torch.tensor([1, 0, 1]).to(float)
    cmap[one_hot_tensor == 3] = torch.tensor([0, 1, 0]).to(float)
    cmap[one_hot_tensor == 4] = torch.tensor([0, 0, 1]).to(float)
    cmap[one_hot_tensor == 5] = torch.tensor([1, 1, 1]).to(float)
    cmap[one_hot_tensor == 6] = torch.tensor([0, 0, 0]).to(float)
    cmap = torch.permute(cmap, (2, 0, 1))
    return cmap


if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    safe_mkdir(args.output)

    model = Deeplabv3(num_classes=7)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    filenames = [file for file in os.listdir(args.input) if file.endswith(".jpg")]

    for i in range(len(filenames)):
        img = Image.open(os.path.join(args.input, filenames[i]))
        img = TF.to_tensor(img)  # [3 * 512 * 512]
        mask = model(
            img[None, :, :, :].to(device)
        )  # expand to [1 * 3 * 512 * 512] input shape is required
        mask = torch.squeeze(mask)  # [7 * 512 * 512]
        pred = from_one_hot_to_rgb(mask.argmax(dim=0))
        name = filenames[i].split(".")[-2]
        save_path = os.path.join(args.output, f"{name}.png")
        save_image(pred, save_path)
