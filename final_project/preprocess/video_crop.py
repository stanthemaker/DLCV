import os
import sys

import cv2
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm.auto import tqdm

sys.path.insert(0, "..")

from setup import ROOT_DIR


def crops_frames_and_borders(
    videopath, newvideopath, starting_frame, ending_frame, bbox, person_id, ttm
):
    cap = cv2.VideoCapture(videopath)
    # player = MediaPlayer(videopath)
    length = int(ending_frame - starting_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

    if cap.isOpened() == False:
        print("Error opening the video file")
    else:
        # print(f"Opening video {videopath}")
        # Get frame rate information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # print("Frame Rate : ", fps, "frames per second")
        # Get frame count
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print("Frame count : ", frame_count)

    # Obtain frame size information using get() method
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    # roi_size = (220, 220)

    # Initialize video writer object
    out = cv2.VideoWriter(
        newvideopath,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        frame_size,
    )

    frames_progression = 0
    # For loop that iterates until the index matches the desired length of the clip
    for i in range(
        length
    ):  # reads through each frome within the loop, and then writes that frame into the new video isolate the roi
        ret, frame = cap.read()
        frame_count = i + starting_frame
        frame_rows = bbox.loc[(bbox["frame_id"] == frame_count)]
        # audio_frame, val = player.get_frame()
        if ret == True:
            # Locating the ROI within the frames that will be cropped
            for index, row in frame_rows.iterrows():
                x1 = int(row["x1"])
                x2 = int(row["x2"])
                y1 = int(row["y1"])
                y2 = int(row["y2"])
                if row["person_id"] == person_id:
                    cv2.putText(
                        frame,
                        str(ttm),
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

            # roi = frame[y_roi : y_roi + box_dim, x_roi : x_roi + box_dim]
            # Write the frame into the file output .avi that was read from the original video in cap.read()
            out.write(frame)
        else:
            print(
                "Cannot retrieve frames. Breaking."
            )  # If a frame cannot be retrieved, this error is thrown
        if out.isOpened() == False:
            print(
                "Error opening the video file"
            )  # If the out video cannot be opened, this error is thrown
        else:
            frames_progression = (
                frames_progression + 1
            )  # Shows how far the frame writting process got. Compare this to the desired frame length
    bbox.drop(bbox[bbox["frame_id"] < ending_frame].index, inplace=True)
    # Release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # These were just for me to verify that the right frames were written
    cap = cv2.VideoCapture(newvideopath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.destroyAllWindows()


video_root = os.path.join(ROOT_DIR, "data")
# i = 0
for train_csv in tqdm(
    os.listdir(os.path.join(video_root, "student_data", "train", "seg")),
    position=0,
    leave=True,
):
    # if i > 217:
    stripped_csv = train_csv.split(".")[0]
    os.makedirs(os.path.join("./processed_video", stripped_csv), exist_ok=True)
    seg = pd.read_csv(
        os.path.join(video_root, "student_data", "train", "seg", train_csv)
    )
    seg.sort_values(by=["end_frame"], inplace=True)
    bbox = pd.read_csv(
        os.path.join(
            video_root,
            "student_data",
            "train",
            "bbox",
            stripped_csv.split("_")[0] + "_bbox.csv",
        )
    )
    bbox.sort_values(by=["frame_id"], inplace=True)
    # if os.path.isdir(os.path.join("./processed_video", stripped_csv)) and len(os.listdir(os.path.join("./processed_video", stripped_csv))) == len(seg):
    #     with open("./log/log_video.txt", "a") as f:
    #             f.write(
    #                 f"skip {stripped_csv}/\n"
    #             )
    #     continue
    # with open("./log/log_video.txt", "a") as f:
    #             f.write(
    #                 f"save at {stripped_csv}/\n"
    #             )
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
            os.path.join("processed_video", stripped_csv, f"{ttm}_{index}_{video_id}"),
            start_frame,
            end_frame,
            bbox,
            person_id,
            ttm,
        )
        
# i+=1

# ffmpeg_extract_subclip(
#     os.path.join(video_root, "student_data", "videos", video_id),
#     start_frame / fps,
#     end_frame / fps,
#     targetname=os.path.join(
#         "processed_video", stripped_csv, f"{ttm}_{index}_{video_id}"
#     ),
# )
