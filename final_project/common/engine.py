import os, logging
import time
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from common.utils import AverageMeter


logger = logging.getLogger(__name__)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    logger.info("training")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()

    model.train()

    end = time.time()

    for i, (video, audio, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)
        if not args.preprocess:
            video = video.to(device)
            audio = audio.to(device)
            target = target.to(device)
        # compute output

        # from common.render import visualize_gaze
        # for i in range(32):
        #     visualize_gaze(video, output[0], index=i, title=str(i))
        # print(video.device)
        # print(audio.device)
        # print(target.device)
        # print(output.device)
            output = model(video, audio)
            loss = criterion(output, target.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                logger.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=avg_loss,
                    )
                )
        else: 
            logger.info(f"preprocess [{i}/{len(train_loader)}]")


def validate(val_loader, model, postprocess, device, args):
    logger.info("evaluating")
    batch_time = AverageMeter()
    model.to(device)
    model.eval()
    end = time.time()

    for i, (video, audio, target) in enumerate(tqdm(val_loader)):

        if not args.preprocess:
            video = video.to(device)
            audio = audio.to(device)

            with torch.no_grad():
                output = model(video, audio)
                postprocess.update(output.detach().cpu(), target)

                batch_time.update(time.time() - end)
                end = time.time()

            if i % 100 == 0:
                logger.info(
                    "Processed: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t".format(
                        i, len(val_loader), batch_time=batch_time
                    )
                )
        else: 
            logger.info(f"preprocess [{i}/{len(val_loader)}]")
    mAP = 0
    if not args.preprocess:
        postprocess.save()
        mAP = postprocess.get_mAP()

    return mAP


def evaluate(val_loader, model, postprocess, device, args):
    logger.info("evaluating")
    batch_time = AverageMeter()
    model.to(device)
    model.eval()
    end = time.time()

    for i, (video, audio, sid) in enumerate(tqdm(val_loader)):
        if not args.preprocess:
            video = video.to(device)
            audio = audio.to(device)

            with torch.no_grad():
                if audio.size(dim=1) == 0:
                    print(sid)
                output = model(video, audio)
                postprocess.update(output.detach().cpu(), sid)

                batch_time.update(time.time() - end)
                end = time.time()

            if i % 100 == 0:
                logger.info(
                    "Processed: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t".format(
                        i, len(val_loader), batch_time=batch_time
                    )
                )
        else:
            logger.info(f"preprocess [{i}/{len(val_loader)}]")
        
    if not args.preprocess:
        postprocess.save()
