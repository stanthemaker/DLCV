import argparse
import random
import torch
import torchvision.utils as vutils
import os
from net import DC_Generator_ML, DC_Generator_origin
from Dataset import UnNormalize
from torch.autograd import Variable
import random


parser = argparse.ArgumentParser()
parser.add_argument("--model1", "-m", type=str, default=None, help="Model path to load")
parser.add_argument("--model2", "-n", type=str, default=None, help="Model path to load")
parser.add_argument(
    "--out_path",
    "-o",
    type=str,
    default="/home/stan/hw2-stanthemaker/problem1_GAN/inference_output",
    help="output path to store generated images",
)
args = parser.parse_args()
output_path = args.out_path
n_outputs = 1000

seed = 1314520
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

state_dict = torch.load(args.model1, map_location=device)
params = state_dict["params"]
generator = DC_Generator_ML(in_dim=params["nz"]).to(device)
generator.load_state_dict(state_dict["generator"])
generator.eval()

state_dict = torch.load(args.model2, map_location=device)
params = state_dict["params"]
generator2 = DC_Generator_origin(params).to(device)
generator2.load_state_dict(state_dict["generator"])
generator2.eval()


unorm = UnNormalize()

# z1 = Variable(torch.randn(n_outputs, params["nz"])).to(device)
# z2 = Variable(torch.randn(n_outputs, params["nz"], 1, 1)).to(device)
# inference_imgs = torch.empty((n_outputs, 3, 64, 64))
with torch.no_grad():
    for i in range(n_outputs):
        if random.random() > 0.5:
            z = Variable(torch.randn(1, params["nz"])).to(device)
            img = generator(z).squeeze().detach().cpu()

        else:
            z = Variable(torch.randn(1, params["nz"], 1, 1)).to(device)
            img = generator2(z).squeeze().detach().cpu()

        img = unorm(img)
        vutils.save_image(img, os.path.join(output_path, f"{i}.png"))
        # *********************
        # *    report      *
        # *********************
    # filename = os.path.join(
    #     "/home/stan/hw2-stanthemaker/problem1_GAN/",
    #     "report_32.png",
    # )
    # vutils.save_image(inference_imgs[:32], filename, nrow=8)
