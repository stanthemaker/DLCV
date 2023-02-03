# $1 : model_1 path ,$2:model_2 path,  $3 : inference output path

python3 /home/stan/hw2-stanthemaker/problem1_GAN/src/inference.py -m $1 -n $2 -o $3 -c cuda:3
# get face score
python3 /home/stan/hw2-stanthemaker/problem1_GAN/src/utils/face_recog.py --image_dir $3
# get FID score
python3 -m pytorch_fid $3 /home/stan/hw2-stanthemaker/hw2_data/face/val --device cuda:3
