import os, sys

sys.path.append("..")
import torch
import torch.optim
import torch.utils.data
from dataset.data_loader import ImagerLoader
from dataset.test_loader import test_ImagerLoader
from dataset.sampler import SequenceBatchSampler
from setup import ROOT_DIR
from torchvision import transforms
import time
from tqdm import tqdm


def collate_fn(batch):
    print("[collate fn] batch length: ", len(batch))
    print("[collate fn] video 0 shape: ", batch[0][0].shape)
    print("[collate fn] audio 0 shape: ", batch[0][1].shape)
    # print(f"[collate fn] labels: ")
    # for b in batch:
    #     print(b[2])

    min_frames = min([b[0].shape[0] for b in batch])
    min_duration = min([b[1].shape[0] for b in batch])
    print("min duration: ", min_duration)
    video = torch.cat([b[0][:min_frames, ...].unsqueeze(0) for b in batch], dim=0)
    # audio = torch.cat([b[1][:, :min_duration] for b in batch], dim=0)
    audio = torch.cat([b[1][:min_duration].unsqueeze(0) for b in batch], dim=0)
    # target = torch.cat([b[2] for b in batch])
    # return video, audio, target
    return video, audio


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def main():
    datapath = os.path.join(ROOT_DIR, "data", "student_data", "test")
    videopath = os.path.join(ROOT_DIR, "data", "student_data", "videos")
    audiopath = "./extracted_audio"
    file_path = "./dataset/split/test.list"
    train_dataset = test_ImagerLoader(
        datapath,
        audiopath,
        videopath,
        file_path,
        # seg_info,
        mode="eval",
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=SequenceBatchSampler(train_dataset, 32),
        num_workers=2,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    for idx, (video, audio, _, _) in enumerate(tqdm(train_loader)):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        # print("[run] iter:", idx, "at ", current_time)
        print("[run] video shape: ", video.shape)
        print("[run] audio shape: ", audio.shape)
        # print("[run] label: ", label)


if __name__ == "__main__":
    main()
