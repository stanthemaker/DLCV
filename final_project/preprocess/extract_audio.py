import os
import sys

import cv2
import pandas as pd
import torchaudio
from moviepy.editor import VideoFileClip
from tqdm.auto import tqdm

sys.path.insert(0, "..")
import argparse

from setup import ROOT_DIR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, help="echo the string you use here")
    parser.add_argument(
        "--interval", type=int, help="how many frame to go forward and backward"
    )
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    video_root = os.path.join(ROOT_DIR, "data")
    seg_path = (
        os.path.join(video_root, "student_data", "train", "seg")
        if args.train
        else os.path.join(video_root, "student_data", "val", "seg")
    )

    for index, csv in enumerate(sorted(os.listdir(seg_path))):
        if index < args.index + 10 and index >= args.index:
            stripped_csv = csv.split(".")[0]
            os.makedirs(os.path.join("./extracted_audio", stripped_csv), exist_ok=True)
            seg = pd.read_csv(os.path.join(seg_path, csv))
            # with open("./log/log.txt", "a") as f:
            #         f.write(
            #             f"{stripped_csv}/\n"
            #         )
            
            for index, row in seg.iterrows():
                start_frame = row["start_frame"]
                end_frame = row["end_frame"]
                ttm = row["ttm"]
                video_id = stripped_csv.split("_")[0] + ".mp4"
                audio_id = stripped_csv.split("_")[0] + ".wav"

                cap = cv2.VideoCapture(
                    os.path.join(video_root, "student_data", "videos", video_id)
                )
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                print("Frame Rate : ", fps, "frames per second")
                # Get frame count
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                start_frame = max(start_frame - args.interval, 0)
                end_frame = min(end_frame + args.interval, frame_count - 1)

                if not args.crop and (f"{ttm}_{index}_{audio_id}") not in os.listdir(
                    os.path.join("./extracted_audio", stripped_csv)
                ):
                    video = VideoFileClip(
                        os.path.join(video_root, "student_data", "videos", video_id)
                    )
                    audio = video.audio
                    audio.write_audiofile(
                        os.path.join(
                            "./extracted_audio",
                            stripped_csv,
                            f"{ttm}_{index}_{audio_id}",
                        )
                    )
                elif (f"{ttm}_{index}_{audio_id}") in os.listdir(
                    os.path.join("./extracted_audio", stripped_csv)
                ) and frame_count > 3000:
                    audio, ori_sample_rate = torchaudio.load(
                        os.path.join(
                            "./extracted_audio",
                            stripped_csv,
                            f"{ttm}_{index}_{audio_id}",
                        )
                    )
                    sample_rate = 16000
                    transform = torchaudio.transforms.Resample(
                        ori_sample_rate, sample_rate
                    )
                    audio = transform(audio)
                    # print(audio.shape)
                    onset = int(start_frame / fps * sample_rate)
                    offset = int(end_frame / fps * sample_rate)
                    print(onset, offset)
                    crop_audio = audio[:, onset:offset]
                    print(crop_audio.shape)
                    torchaudio.save(
                        os.path.join(
                            "./extracted_audio",
                            stripped_csv,
                            f"{ttm}_{index}_{audio_id}",
                        ),
                        crop_audio,
                        sample_rate,
                    )
                    


if __name__ == "__main__":
    os.makedirs(os.path.join("./extracted_audio"), exist_ok=True)
    args = get_args()
    main(args)
