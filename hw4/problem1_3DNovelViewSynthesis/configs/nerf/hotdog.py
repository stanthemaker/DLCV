_base_ = "../default.py"

expname = "dvgo_hotdog"
basedir = (
    "/home/stan/hw4-stanthemaker/problem1_3DNovelViewSynthesis/logs/nerf_synthetic"
)

data = dict(
    datadir="/home/stan/hw4-stanthemaker/hw4_data/hotdog",
    dataset_type="blender",
    white_bkgd=True,
)
