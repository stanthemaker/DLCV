# $1 : model path , $2 : inference output path

python3 /home/stan/hw2-stanthemaker/problem1_GAN/src/generate.py -m $1 -o $2 -c cuda:3
# get face score
python3 /home/stan/hw2-stanthemaker/problem1_GAN/src/utils/face_recog.py --image_dir $2
# get FID score
python3 -m pytorch_fid $2 /home/stan/hw2-stanthemaker/hw2_data/face/val --device cuda:3