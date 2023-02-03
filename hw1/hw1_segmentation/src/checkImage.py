import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
import sys
import os
import random
from net import Deeplabv3, FCN32_VGG16
from torchvision.transforms import functional as TF


def from_one_hot_to_rgb(one_hot_tensor: torch.Tensor, name: str) -> None:
    """Assign a different color to each class in the input tensor"""
    print("name:", name)
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
    save_image(cmap, f"{name}.png")


path = "../p2_data/validation/"
images = sorted(
    [
        os.path.join(path, x)
        for x in os.listdir(path)
        if x.endswith("0013_sat.jpg")
        or x.endswith("0062_sat.jpg")
        or x.endswith("0104_sat.jpg")
    ]
)
device = "cuda:3"
torch.cuda.set_device(device)
# model
if len(sys.argv) <= 1:
    print("Missing argument 1")
if sys.argv[1] == "--Deeplabv3":
    model = Deeplabv3(num_classes=7)
elif sys.argv[1] == "--FCN32":
    # TODO: implement vgg16-fcn8
    model = FCN32_VGG16()
else:
    print("Wrong argument 1, exit")
    sys.exit()
model = model.to(device)
if len(sys.argv) == 3:
    try:
        model.load_state_dict(torch.load(sys.argv[2]))
    except:
        print("Wrong ckpt path, exit")
        sys.exit()
        
model = model.to(device)
batch = torch.empty(3, 3, 512, 512)
for i in range(len(images)):
    img = Image.open(images[i])
    img = TF.to_tensor(img)
    batch[i] = img

mask = model(batch.to(device))["out"]
for i in range(len(mask)):
    from_one_hot_to_rgb(mask[i].argmax(dim=0), images[i].split("/")[-1].split(".")[0])
