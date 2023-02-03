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
parser.add_argument("-model1", "-m", type=str, default=None, help="Model path to load")
parser.add_argument("-model2", "-n", type=str, default=None, help="Model path to load")
parser.add_argument(
    "-out_path",
    "-o",
    type=str,
    default="/home/stan/hw2-stanthemaker/problem1_GAN/inference_output",
    help="output path to store generated images",
)
parser.add_argument("-cuda", "-c", default="cuda", help="choose a cuda")
args = parser.parse_args()
output_path = args.out_path
n_outputs = 1000

seed = 1314520
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)
device = torch.device(args.cuda)

state_dict = torch.load(args.model1)
params = state_dict["params"]
generator = DC_Generator_ML(in_dim=params["nz"]).to(device)
generator.load_state_dict(state_dict["generator"])
generator.eval()

state_dict = torch.load(args.model2)
params = state_dict["params"]
generator2 = DC_Generator_origin(params).to(device)
generator2.load_state_dict(state_dict["generator"])
generator2.eval()


unorm = UnNormalize()

z1 = Variable(torch.randn(n_outputs, params["nz"])).to(device)
z2 = Variable(torch.randn(n_outputs, params["nz"], 1, 1)).to(device)
inference_imgs = torch.empty((n_outputs, 3, 64, 64))
with torch.no_grad():
    imgs_1 = generator(z1).detach().cpu()
    imgs_2 = generator2(z2).detach().cpu()
    for i in range(n_outputs):
        if random.random() > 0.5:
            inference_imgs[i] = unorm(imgs_1[i])
        else:
            
            inference_imgs[i] = unorm(imgs_2[i])
        vutils.save_image(inference_imgs[i], os.path.join(output_path, f"{i}.png"))
        # *********************
        # *    report      *
        # *********************
    filename = os.path.join(
        "/home/stan/hw2-stanthemaker/problem1_GAN/",
        "report_32.png",
    )
    vutils.save_image(inference_imgs[:32], filename, nrow=8)
