#!/bin/bash
wget -O ./pa2.pt "https://www.dropbox.com/s/dcca53dmj8rq5sx/finetune_1217-1533_C_224.pt?dl=1"
# model="/home/stan/hw4-stanthemaker/problem2_SSL/ckpt/finetune_1217-1533_C_224.pt"
python3 ./problem2_SSL/src/inference.py --input $1 --img_dir $2 --output $3 --model ./pa2.pt
# TODO - run your inference Python3 code