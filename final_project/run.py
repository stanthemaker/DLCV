import os, sys, random, pprint

sys.path.append(".")
import json
import torch
import soundfile as sf
import torch.optim
import torch.utils.data

# import soundfile as sf
from tqdm import tqdm
from dataset.data_loader import ImagerLoader, test_ImagerLoader

# from preprocess.dataset.test_loader import test_ImagerLoader
from dataset.sampler import SequenceBatchSampler
from model.model import BaselineLSTM, ViViT
from config import argparser
from common.logger import create_logger
from common.engine import train, validate, evaluate
from common.utils import (
    PostProcessor,
    test_PostProcessor,
    get_transform,
    save_checkpoint,
    collate_fn,
)
from setup import ROOT_DIR


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(args.device_id)
        torch.cuda.init()
    if not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)

    logger, timestamp = create_logger(args)
    pp = pprint.PrettyPrinter(indent=2)
    logger.info(pprint.pformat(vars(args)))
    logger.info(f"Model: {args.model}")
    logger.info(f"Using device: {device}")
    logger.info("preprocess" if args.preprocess else "not preprocess")
    
    model = None
    if args.model == "BaselineLSTM":
        model = BaselineLSTM(args).to(device)
    else:
        model = ViViT(args, device, num_frames=args.maxframe + 2).to(device)
        
        # If it is vivit, use our own checkpoint loader / BaselineLSTM -> Built in loader
        if not args.checkpoint == "None":
            logger.info(f"loading model {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint)["state_dict"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    videopath = os.path.join(ROOT_DIR, "data", "student_data", "videos")
    train_loader = None

    
    
    if not args.eval:
        datapath = os.path.join(ROOT_DIR, "data", "student_data", "train")
        train_dataset = ImagerLoader(
            datapath,
            videopath,
            args.train_file,
            args.maxframe,
            args.minframe,
            mode="train",
            transform=get_transform(True),
            img_size=args.img_size,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=SequenceBatchSampler(train_dataset, args.batch_size, shuffle=True),
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )

        class_weights = torch.FloatTensor(args.weights).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(model.parameters(), args.lr)

        val_dataset = ImagerLoader(
            datapath,
            videopath,
            args.val_file,
            args.maxframe,
            args.minframe,
            mode="val",
            transform=get_transform(False),
            img_size=args.img_size,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=False
        )
    else:
        datapath = os.path.join(ROOT_DIR, "data", "student_data", "test")
        test_dataset = test_ImagerLoader(
            datapath,
            videopath,
            args.maxframe,
            args.minframe,
            mode="test",
            transform=get_transform(False),
            img_size=args.img_size,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=False,
        )

    best_mAP = 0

    if not args.eval:
        logger.info("start training")
        
        for epoch in range(args.epochs):

            train_loader.batch_sampler.set_epoch(epoch)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, device, args)

            # save a version before validating
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "mAP": 0},
                save_path=args.exp_path,
                is_best=False,
                timestamp=timestamp,
            )
            
            # TODO: implement distributed evaluation
            # evaluate on validation set
            postprocess = PostProcessor(args)
            mAP = validate(val_loader, model, postprocess, device, args)

            # remember best mAP in validation and save checkpoint
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            logger.info(f"mAP: {mAP:.4f} best mAP: {best_mAP:.4f}")
            
            # save a version after getting mAP score
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "mAP": mAP},
                save_path=args.exp_path,
                is_best=is_best,
                timestamp=timestamp,
            )

    else:        
        logger.info("start evaluating")
        postprocess = test_PostProcessor(args)
        mAP = evaluate(test_loader, model, postprocess, device, args)
        print("Score mAP:", mAP)
        postprocess.mkfile()


# def get_segInfo():
#     args = argparser.parse_args()
#     frames_dir = "./extracted_frames"
#     aud_dir = "./extracted_audio"
#     seginfo = {}

#     for seg_id in tqdm(os.listdir(frames_dir)):
#         frame_dir = os.path.join(frames_dir, seg_id)
#         seginfo[seg_id] = {}
#         frame_list = []
#         for f in os.listdir(os.path.join(frame_dir)):
#             fid = int(f.split("_")[1])
#             # fid = f.split(".")[0]
#             frame_list.append(fid)
#         frame_list.sort()
#         seginfo[seg_id]["frame_list"] = frame_list
#         aud, sr = sf.read(os.path.join(aud_dir, f"{seg_id}.wav"))
#         frame_num = int(aud.shape[0] / sr * 30 + 1)
#         frame_list_len = 0
#         try:
#             frame_list_len = max(frame_list)
#         except:
#             pass
#         seginfo[seg_id]["frame_num"] = max(frame_num, frame_list_len + 1)
#         # seginfo[seg_id]["frame_num"] = max(frame_num, len(frame_list) + 1)

#     with open("./seg_info.json", "w") as f:
#         json.dump(seginfo, f, indent=4)
# main(args)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)
