# import argparse
# import random
# import torch
# import torchvision.utils as vutils
# import os
# from net import EnsembledModel
# from Dataset import UnNormalize


# parser = argparse.ArgumentParser()
# parser.add_argument("--model1", "-m", type=str, default=None, help="model1")
# parser.add_argument(
#     "--model2", "-n", default=None, type=str, help="model2"
# )
# parser.add_argument(
#     "--out_path",
#     "-o",
#     type=str,
#     default="/home/stan/hw2-stanthemaker/problem1_GAN/inference_output",
#     help="output path to store generated images",
# )
# parser.add_argument("-cuda", "-c", default="cuda", help="choose a cuda")
# args = parser.parse_args()
# n_outputs = 1000
# output_path = args.out_path

# seed = 1314520
# random.seed(seed)
# torch.manual_seed(seed)
# print("Random Seed: ", seed)
# device = torch.device(args.cuda)

# state_dict1 = torch.load(args.model1)
# state_dict2 = torch.load(args.model2)
# params1 = state_dict1["params"]
# params2 = state_dict2["params"]

# generator = EnsembledModel(params1 , state_dict1["generator"] , params2 , state_dict2["generator"], device)
# unorm = UnNormalize()

# z = torch.randn(n_outputs, params1["nz"], device=device)
# sample_noise = z[:32]

# with torch.no_grad():
#     imgs = generator(z).detach().cpu()
#     for i in range(n_outputs):
#         vutils.save_image(unorm(imgs[i]), os.path.join(output_path, f"{i}.png"))
#     #     # *********************
#     #     # *    report      *
#     #     # *********************
#     f_imgs_sample = (imgs[:32].data + 1) / 2.0
#     filename = os.path.join(
#         "/home/stan/hw2-stanthemaker/problem1_GAN/",
#         "report_32.png",
#     )
#     vutils.save_image(f_imgs_sample, filename, nrow=8)
