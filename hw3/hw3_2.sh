#!/bin/bash

# TODO - run your inference Python3 code
python3 problem2_ImageCaptioning/src/inference.py -t ./caption_tokenizer.json -m ./pa2.ckpt -i $1 -j $2