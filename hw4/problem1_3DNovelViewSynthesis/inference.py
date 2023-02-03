import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from inference_tools import load_blender_data
from torch_efficient_distloss import flatten_eff_distloss


def config_parser():
    """Define command line arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, help="config file path")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")
    parser.add_argument(
        "--ft_path",
        type=str,
        default="",
        help="specific weights npy file to reload for fine network",
    )
    parser.add_argument(
        "--ct_path",
        type=str,
        default="",
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--testsavedir",
        type=str,
        default="",
        help="path to dump images",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="",
        help="path to transform_test.json",
    )
    parser.add_argument("--dump_images", action="store_true")
    parser.add_argument("--eval_ssim", action="store_true")
    parser.add_argument("--eval_lpips_alex", action="store_true")
    parser.add_argument("--eval_lpips_vgg", action="store_true")

    return parser


def seed_everything():
    """Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


@torch.no_grad()
def render_viewpoints(
    model,
    render_poses,
    HW,
    Ks,
    ndc,
    render_kwargs,
    gt_imgs=None,
    savedir=None,
    dump_images=False,
    filenames=[],
    render_factor=0,
    render_video_flipy=False,
    render_video_rot90=0,
    eval_ssim=False,
    eval_lpips_alex=False,
    eval_lpips_vgg=False,
):
    """Render images for the given viewpoints; run evaluation if gt given."""
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW / render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H,
            W,
            K,
            c2w,
            ndc,
            inverse_y=render_kwargs["inverse_y"],
            flip_x=cfg.data.flip_x,
            flip_y=cfg.data.flip_y,
        )
        keys = ["rgb_marched", "depth", "alphainv_last"]
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(
                rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0)
            )
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result["rgb_marched"].cpu().numpy()
        depth = render_result["depth"].cpu().numpy()
        bgmap = render_result["alphainv_last"].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        # if i == 0:
        #     print("Testing", rgb.shape)

        # if gt_imgs is not None and render_factor == 0:
        #     p = -10.0 * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
        #     psnrs.append(p)
    #         if eval_ssim:
    #             ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
    #         if eval_lpips_alex:
    #             lpips_alex.append(
    #                 utils.rgb_lpips(rgb, gt_imgs[i], net_name="alex", device=c2w.device)
    #             )
    #         if eval_lpips_vgg:
    #             lpips_vgg.append(
    #                 utils.rgb_lpips(rgb, gt_imgs[i], net_name="vgg", device=c2w.device)
    #             )

    # if len(psnrs):
    #     print("Testing psnr", np.mean(psnrs), "(avg)")
    #     if eval_ssim:
    #         print("Testing ssim", np.mean(ssims), "(avg)")
    #     if eval_lpips_vgg:
    #         print("Testing lpips (vgg)", np.mean(lpips_vgg), "(avg)")
    #     if eval_lpips_alex:
    #         print("Testing lpips (alex)", np.mean(lpips_alex), "(avg)")

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0, 1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0, 1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0, 1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, f"{filenames[i]}.png")
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def load_data_dict(args, cfg):
    K, depths = None, None
    near_clip = None
    poses, render_poses, hwf, i_split, filenames = load_blender_data(
        args.json_path, False, 1
    )

    i_test = i_split
    near, far = 2.0, 6.0
    args.white_bkgd = True
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([[800, 800] for _ in poses])
    # if images.shape[-1] == 4:
    #     if args.white_bkgd:
    #         images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    #     else:
    #         images = images[..., :3] * images[..., -1:]
    # HW = np.array([im.shape[:2] for im in images])
    # irregular_shape = images.dtype is np.dtype("object")

    if K is None:
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[..., :4]
    data_dict = dict(
        hwf=hwf,
        HW=HW,
        Ks=Ks,
        near=near,
        far=far,
        near_clip=near_clip,
        i_test=i_test,
        poses=poses,
        render_poses=render_poses,
        # images=images,
        depths=depths,
        # irregular_shape=irregular_shape,
        filenames=filenames,
    )

    # print(data_dict["poses"].shape) = (50,4,4)
    # print(data_dict["poses"][data_dict["i_test"]].shape) = (1, 50,4,4)
    kept_keys = {
        "hwf",
        "HW",
        "Ks",
        "near",
        "far",
        "near_clip",
        "i_train",
        "i_val",
        "i_test",
        # "irregular_shape",
        "poses",
        "render_poses",
        # "images",
        "filenames",
    }
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    data_dict["poses"] = torch.Tensor(data_dict["poses"])
    return data_dict


if __name__ == "__main__":

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    seed_everything()
    print("---------------- loading --------------------")
    data_dict = load_data_dict(args=args, cfg=cfg)
    ckpt_path = args.ft_path
    model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, ckpt_path, args.ct_path).to(device)
    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        "model": model,
        "ndc": cfg.data.ndc,
        "render_kwargs": {
            "near": 2.0,
            "far": 6.0,
            "bg": 1 if cfg.data.white_bkgd else 0,
            "stepsize": stepsize,
            "inverse_y": cfg.data.inverse_y,
            "flip_x": cfg.data.flip_x,
            "flip_y": cfg.data.flip_y,
            "render_depth": True,
        },
    }

    testsavedir = args.testsavedir
    os.makedirs(testsavedir, exist_ok=True)
    print("All results are dumped into", testsavedir)
    rgbs, depths, bgmaps = render_viewpoints(
        render_poses=data_dict["poses"][data_dict["i_test"]],
        # HW=data_dict["HW"][data_dict["i_test"]],
        HW=data_dict["HW"],
        Ks=data_dict["Ks"],
        gt_imgs=None,
        savedir=testsavedir,
        dump_images=args.dump_images,
        filenames=data_dict["filenames"],
        eval_ssim=args.eval_ssim,
        eval_lpips_alex=args.eval_lpips_alex,
        eval_lpips_vgg=args.eval_lpips_vgg,
        **render_viewpoints_kwargs,
    )


print("Done")
