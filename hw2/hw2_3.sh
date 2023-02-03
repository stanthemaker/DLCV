#!/bin/bash
python3 ./problem3_DANN/src/inference.py -d $1 -s ./svhn.ckpt -u ./usps.ckpt -o $2
# python3 digit_classifier.py --folder /content/output