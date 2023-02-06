import os
import sys

import cv2
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm.auto import tqdm

sys.path.insert(0, "..")

from setup import ROOT_DIR


def crops_frames_and_borders(
    videopath, newvideopath, starting_frame, ending_frame, person_id, ttm
):
    cap = cv2.VideoCapture(videopath)
    # player = MediaPlayer(videopath)
    length = int(ending_frame - starting_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

    if cap.isOpened() == False:
        print("Error opening the video file")
    else:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)



    for i in range(
        length
    ):  # reads through each frome within the loop, and then writes that frame into the new video isolate the roi
        ret, frame = cap.read()
        frame_count = i + starting_frame
        if not os.path.isfile(os.path.join(newvideopath, f"img_{frame_count}.png")) and ret==True:
            cv2.imwrite(os.path.join(newvideopath, f"img_{frame_count:05d}.png"), frame, [cv2.IMWRITE_PNG_COMPRESSION, 7])
            print(f"img_{frame_count}.png")
        # audio_frame, val = player.get_frame()




def thread():
    video_root = os.path.join(ROOT_DIR, "data")
    for train_csv in tqdm(
        sorted(os.listdir(os.path.join(video_root, "student_data", "train", "seg"))),
        position=0,
        leave=True,
    ):
        stripped_csv = train_csv.split(".")[0]
        uid = stripped_csv.split("_")[0]

        os.makedirs(os.path.join("./compressed_frames", uid), exist_ok=True)
        seg = pd.read_csv(
            os.path.join(video_root, "student_data", "train", "seg", train_csv)
        )
        seg.sort_values(by=["end_frame"], inplace=True)

        print(stripped_csv)
        for index, row in seg.iterrows():

            start_frame = row["start_frame"]
            end_frame = row["end_frame"]
            ttm = row["ttm"]
            person_id = row["person_id"]

            video_id = stripped_csv.split("_")[0] + ".mp4"
            cap = cv2.VideoCapture(
                os.path.join(video_root, "student_data", "videos", video_id)
            )
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # Get frame count
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            start_frame = max(start_frame - 30, 0)

            end_frame = min(end_frame + 30, frame_count - 1)
            crops_frames_and_borders(
                os.path.join(video_root, "student_data", "videos", video_id),
                os.path.join("compressed_frames", uid),
                start_frame,
                end_frame,
                person_id,
                ttm,
            )
        length = len(os.listdir(os.path.join("compressed_frames", uid)))
        with open("./log/log_video.txt", "a") as f:
                    f.write(
                        f"{uid}: {str(length)}\n"
                    )
        print(uid, len(os.listdir(os.path.join("compressed_frames", uid))))

if __name__ == "__main__":
    thread()