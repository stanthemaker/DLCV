#!/bin/bash

# TODO - run your inference Python3 code
python3 ./hw1_segmentation/src/evaluate.py --input $1 --output $2 --model ./Deeplabv3.ckpt