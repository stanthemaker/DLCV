#!/bin/bash
# python3 run.py $1
# python3 ./problem1_3DNovelViewSynthesis/run.py --config ./problem1_3DNovelViewSynthesis/configs/nerf/hotdog.py --ft_path  --render_only --inference --dump_images --testsavedir $1
# ft_path="./problem1_3DNovelViewSynthesis/logs/nerf_synthetic/dvgo_hotdog/fine_last.tar"
# ct_path="./problem1_3DNovelViewSynthesis/logs/nerf_synthetic/dvgo_hotdog/coarse_last.tar"
cfg_path="./problem1_3DNovelViewSynthesis/configs/nerf/hotdog.py"
wget -O ./coarse.tar "https://www.dropbox.com/s/r1j4vez5jwimlw5/coarse_last.tar?dl=1"
wget -O ./fine.tar "https://www.dropbox.com/s/p06orm3nr6nfu2u/fine_last.tar?dl=1"

python3 ./problem1_3DNovelViewSynthesis/inference.py --config $cfg_path --ft_path ./fine.tar --ct_path ./coarse.tar --json_path $1 --testsavedir $2 --dump_images
# python3 ./grade.py ./problem1_3DNovelViewSynthesis/logs/nerf_synthetic/dvgo_hotdog/inference ./hw4_data/hotdog/val