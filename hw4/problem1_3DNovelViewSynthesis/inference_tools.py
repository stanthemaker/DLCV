import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
    )
    return c2w


def load_blender_data(json_path, half_res=False, testskip=1):
    splits = ["test"]
    metas = {}
    for s in splits:
        with open(json_path, "r") as fp:
            metas[s] = json.load(fp)
    # basedir = "/home/stan/hw4-stanthemaker/hw4_data/hotdog"

    all_imgs = []
    all_poses = []
    counts = [0]
    filenames = []
    meta = metas[s]
    poses = []
    if testskip == 0:
        skip = 1
    else:
        skip = testskip
    for frame in meta["frames"][::skip]:
        filenames.append(frame["file_path"].split("/")[-1])
        poses.append(np.array(frame["transform_matrix"]))
        # imgs.append(imageio.imread(fname))
        # fname = os.path.join(basedir, frame["file_path"] + ".png")

    poses = np.array(poses).astype(np.float32)
    all_poses.append(poses)
    i_split = [np.arange(0, len(filenames)) for i in range(len(splits))]
    # imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    # H, W = imgs[0].shape[:2]
    H, W = 800, 800
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 160 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0
    return poses, render_poses, [H, W, focal], i_split, filenames
