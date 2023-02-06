# DLCV Final Project ( Talking to me )
# Important note:
I have softlinks to these directories: **extracted_frames, extracted_audio, data**
Please pay attention to them when fetching this version

# Installation
```bash script=
$ pip install -r requirements.txt
```
- install best.pth at [checkpoint](https://www.dropbox.com/scl/fo/r12wev6kbfghd2wkzqwne/h?dl=0&rlkey=i1k06jo5yqzahyvlg2ssmqdde)

# Preprocess
**Need to preprocess first before inference and train**

**Please strictly follow the file structure**

*For All*
```shell script=
bash preprocess.sh
```

**For the following two commands, we recommand you to run them twice to make sure there is no saving error**.     
- *For Training videos*
```bash script=
$ python run.py --preprocess 
```

- *For Testing videos*
```bash script=
$ python run.py --preprocess --eval
```


# How to inference 
Make sure you preprocess the data
To inference a csv, you need to run the following command:
```bash script= 
$ python3 run.py --eval --checkpoint <checkpoint path> --model <model name - BaselineLSTM or ViViT> --num_worker <option> --device_id <option> 
```
For our best model, please set `--img_size 224 --maxframe 398`  

Output csv file will be located in the following **./evalai_test/output/<exp_name>/results/pred.csv**
To specify the threshold, you may check this file **./evalai_test/threhold.py**
```bash script= 
$ python ./evalai_test/threshold.py --threshold <threshold value> --input_file <path to input file> 

```

# How to Train
Make sure you preprocess the data
To train a model, you can run:
```bash script= 
$ python3 run.py  --num_worker <option> --device_id <option> --model <model name - BaselineLSTM or ViViT> --batch_size <batch size> --img_size <image size> --maxframe <maxframe>
```


# File structure
```bash
├── common
│   ├── engine.py
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   ├── render.py
│   └── utils.py
├── config.py
├── data 
│   └── student_data
│       ├── train
│       ├── test
│       └── videos
├── dataset
│   ├── data_loader.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── sampler.py
│   └── split
│       ├── test.list
│       ├── train.list
│       └── val.list
├── environment.yml
├── evalai_test
│   ├── checkpoint
│   │   └── tmp.txt
│   ├── output
│   │   ├── BaselineLSTM
│   │   │   ├── gt.csv.rank.0
│   │   │   ├── pred.csv.rank.0
│   │   │   └── result
│   │   │       ├── gt.csv
│   │   │       └── pred.csv
│   │   ├── checkpoint
│   │   │   └── tmp.txt
│   │   ├── log
│   │   ├── result
│   │   │   ├── gt.csv
│   │   │   ├── pred.csv
│   │   │   └── threshold.py 
│   │   ├── tmp
│   │   │   └── pred_0.csv
│   │   └── ViViT
│   │       └── result
│   │           ├── gt.csv
│   │           └── pred.csv
│   └── ttm-best.pth
├── extracted_audio 
├── extracted_frames 
├── model
│   ├── __init__.py
│   ├── model.py
│   ├── module.py
│   ├── __pycache__
│   ├── resnet.py
│   └── resse.py
├── preprocess
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── sampler.py
│   │   └── split
│   │       ├── test.list
│   │       ├── train.list
│   │       └── val.list
│   ├── extract_audio.py
│   ├── extracted_audio 
│   ├── extracted_frames 
│   ├── extract_frames.py
│   ├── run.py
│   └── video_crop.py
├── __pycache__
├── README.md
├── requirements.txt
├── run.py
├── scripts
│   ├── download_clips.py
│   ├── get_json.py
│   ├── get_lam_result.py
│   ├── get_ttm_result.py
│   ├── merge.py
│   └── preprocessing.py
├── setup.py
└── test.py
```
# How to run your code?
Bash is not thoroughly tested. If possible, use the python commands above.    
**Need to preprocess first before inference and train**   
**Please strictly follow the file structure**   
 
```shell script=
bash preprocess.sh
```

## Training
```shell script=
bash train.sh 
```
## Inferencing
```shell script=
bash inference.sh <Path to checkpoint>
```


# Submission Rules
### Deadline
111/12/29 (Thur.) 23:59 (GMT+8)
For more details, please click [this link](https://docs.google.com/presentation/d/1Y-gwBmucYgbWLLk-u6coHi7LybFLXgA9gV8KiOiKShI/edit?usp=sharing) to view the slides of Final Project - Talking to me. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion

