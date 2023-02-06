import glob
import json
import logging
import os
import random
from collections import OrderedDict, defaultdict
import cv2
import numpy as np
import pandas as pd
import soundfile
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
from moviepy.editor import VideoFileClip
import torchaudio
import time
logger = logging.getLogger(__name__)


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def helper():
    return defaultdict(OrderedDict)


def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
        x = frame["x"]
        y = frame["y"]
        w = frame["width"]
        h = frame["height"]
        if (
            w <= 0
            or h <= 0
            or frame["frameNumber"] == 0
            or len(frame["Person ID"]) == 0
        ):
            continue
        framenum.append(frame["frameNumber"])
        x = max(x, 0)
        y = max(y, 0)
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)

    if len(framenum) == 0:
        return inter_track

    framenum = np.array(framenum)
    bboxes = np.array(bboxes)

    gt_frames = framenum[-1] - framenum[0] + 1

    frame_i = np.arange(framenum[0], framenum[-1] + 1)

    if gt_frames > framenum.shape[0]:
        bboxes_i = []
        for ij in range(0, 4):
            interpfn = interp1d(framenum, bboxes[:, ij])
            bboxes_i.append(interpfn(frame_i))
        bboxes_i = np.stack(bboxes_i, axis=1)
    else:
        frame_i = framenum
        bboxes_i = bboxes

    # assemble new tracklet
    template = track[0]
    for i, (frame, bbox) in enumerate(zip(frame_i, bboxes_i)):
        record = template.copy()
        record["frameNumber"] = frame
        record["x"] = bbox[0]
        record["y"] = bbox[1]
        record["width"] = bbox[2] - bbox[0]
        record["height"] = bbox[3] - bbox[1]
        inter_track.append(record)
    return inter_track


def normalize(samples, desired_rms=0.1, eps=1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples


def get_bbox(bbox_path):
    bboxes = {}
    bbox_csv = pd.read_csv(bbox_path)
    for idx, bbox in bbox_csv.iterrows():

        # check the bbox, interpolate when necessary
        # frames = check(frames)

        # for frame in frames:
        frameid = int(bbox["frame_id"])
        personid = int(bbox["person_id"])
        bbox = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
        identifier = str(frameid) + ":" + str(personid)
        bboxes[identifier] = bbox

    return bboxes


def make_dataset(file_list, data_path, maxframe, minframe, mode):
    # file list is a list of training or validation file names

    face_crop = {}
    segments = []
    for uid in file_list:
        seg_path = os.path.join(data_path, "seg", uid + "_seg.csv")
        bbox_path = os.path.join(data_path, "bbox", uid + "_bbox.csv")
        uid = uid.strip()
        face_crop[uid] = get_bbox(bbox_path)
        seg = pd.read_csv(seg_path)

        for idx, gt in seg.iterrows():
            personid = gt["person_id"]
            label = int(gt["ttm"])
            start_frame = int(gt["start_frame"])
            end_frame = int(gt["end_frame"])
            # logger.info(f"{uid} -> {start_frame} and {end_frame}")
            seg_length = end_frame - start_frame + 1

            ##### for setting maximum frame size and minimum frame size
            if (
                (mode == "train" and minframe != None and seg_length < minframe)
                or (seg_length <= 1)
                or (personid == 0)
            ):
                continue
            elif maxframe != None and seg_length > maxframe:
                it = int(seg_length / maxframe)
                for i in range(it):
                    sub_start = start_frame + i * maxframe
                    sub_end = min(end_frame, sub_start + maxframe)
                    sub_length = sub_end - sub_start + 1
                    if minframe != None and sub_length < minframe:
                        continue
                    segments.append([uid, personid, label, sub_start, sub_end, idx])
            else:
                
                segments.append([uid, personid, label, start_frame, end_frame, idx])

    return segments, face_crop


def makeFileList(filepath):
    with open(filepath, "r") as f:
        videos = f.readlines()
    return [uid.strip() for uid in videos]


class ImagerLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        video_path,
        file_path,
        maxframe,
        minframe,
        mode="train",
        transform=None,
        img_size=128,
    ):
        self.audio_path = "./extracted_audio"
        self.video_path = video_path
        self.file_path = file_path
        self.file_list = makeFileList(self.file_path)
        print(f"{mode} file with length: {str(len(self.file_list))}")

        self.mode = mode
        segments, face_crop = make_dataset(
            self.file_list, data_path, maxframe, minframe, self.mode
        )
        print("finish making dataset")
        self.segments = segments
        self.face_crop = face_crop
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, indices):
        source_video = self._get_video(indices)
        source_audio = self._get_audio(indices)
        target = self._get_target(indices)
        return source_video, source_audio, target

    def __len__(self):
        return len(self.segments)

    def _get_video(self, index, debug=False):
        video_size = self.img_size
        uid, personid, _, start_frame, end_frame, _ = self.segments[index]
        # logger.info(f"[get audio] {start_frame} and {end_frame}")
        # uid = "be99e9e2-82f7-4f1f-aff5-371834fde0da"
        # start_frame = 3052
        # end_frame = 3082
        cap = cv2.VideoCapture(os.path.join(self.video_path, f"{uid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video = []
        for i in range(start_frame, end_frame + 1):
            key = str(i) + ":" + str(personid)
            # print("key: ", key)
            # print("face: ", dict(list(self.face_crop[uid].items())[:3]))
            if key in self.face_crop[uid].keys():
                bbox = self.face_crop[uid][key]
                if os.path.isfile(
                    f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png"
                ):
                    img = cv2.imread(
                        f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png"
                    )
                    face = cv2.resize(img, (video_size, video_size))
                else:
                    ret, img = cap.read()

                    if not ret:
                        print("not ret")
                        video.append(
                            np.zeros((1, video_size, video_size, 3), dtype=np.uint8)
                        )
                        continue

                    if not os.path.isdir(f"./extracted_frames/{uid}"):
                        os.mkdir(f"./extracted_frames/{uid}")
                    x1, y1, x2, y2 = (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    )

                    face = img[y1:y2, x1:x2, :]
                    if face.size != 0:
                        print(f"{uid}/write: {i:05d}_{personid}")
                        cv2.imwrite(
                            f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png", face
                        )
                        logger.info(f"writing {uid}/img_{i:05d}_{personid}.png")
                try:
                    face = cv2.resize(face, (video_size, video_size))
                except:
                    # bad bbox
                    face = np.zeros((video_size, video_size, 3), dtype=np.uint8)

                if debug:
                    import matplotlib.pyplot as plt

                    plt.imshow(face)
                    plt.show()

                video.append(np.expand_dims(face, axis=0))
            else:
                print("not in face crop")
                video.append(np.zeros((1, video_size, video_size, 3), dtype=np.uint8))
                continue
        cap.release()
        video = np.concatenate(video, axis=0)
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)
        # print("[get video] video shape: ", video.shape)
        return video

    def _get_audio(self, index):
        uid, _, _, start_frame, end_frame, _ = self.segments[index]
        # logger.info(f"[get audio] {start_frame} and {end_frame}")
        # uid = "be99e9e2-82f7-4f1f-aff5-371834fde0da"

        if not os.path.isfile(os.path.join(self.audio_path, f"{uid}.wav")):
            video = VideoFileClip(os.path.join(self.video_path, f"{uid}.mp4"))
            audio = video.audio
            audio.write_audiofile(os.path.join(self.audio_path, f"{uid}.wav"))
            logger.info(f"writing {self.audio_path}/{uid}.wav")
            
       
        audio, sample_rate = torchaudio.load(
            f"{self.audio_path}/{uid}.wav", normalize=True
        )
        
            
    
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = transform(audio)
        # transform = torchaudio.transforms.DownmixMono(channels_first=True)
        # audio = transform(audio)
        audio = torch.mean(audio, dim=0)
        # logger.info(f"[get audio] orig audio shape {audio.shape}")

        onset = int(start_frame / 30 * 16000)
        offset = int(end_frame/ 30 * 16000)
        # onset = int(max((start_frame - 30) / 30 * 16000, 0))
        # offset = int(min((end_frame + 30) / 30 * 16000, 4799999))
        
        crop_audio = audio[onset:offset]
        
        if crop_audio.size()[0] == 0:
            logger.info(f"{onset} and {offset}")
            logger.info(f"{start_frame} and {end_frame}")
            logger.info(f"[get audio] crop audio shape {crop_audio.shape} {uid}")
            video = VideoFileClip(os.path.join(self.video_path, f"{uid}.mp4"))
            audio = video.audio
            audio.write_audiofile(os.path.join(self.audio_path, f"{uid}.wav"))
            
        # if self.mode == 'eval':
        # l = offset - onset
        # crop_audio = np.zeros(l)
        #     index = random.randint(0, len(self.segments)-1)
        #     uid, _, _, _, _, _ = self.segments[index]
        #     audio, sample_rate = soundfile.read(f'{self.audio_path}/{uid}.wav')
        #     crop_audio = normalize(audio[onset: offset])
        # else:
        #     crop_audio = normalize(audio[onset: offset])
        return crop_audio.to(torch.float32)

    def _get_target(self, index):
        if self.mode == "train":
            return torch.LongTensor([self.segments[index][2]])
        else:
            return self.segments[index]


def testmakeFileList(datapath):
    segids = os.listdir(os.path.join(datapath, "seg"))
    return [x.split("_")[0] for x in segids]


def test_make_dataset(file_list, data_path, maxframe=None, minframe=10, mode="eval"):
    # file list is a list of training or validation file names

    face_crop = {}
    segments = []

    for uid in tqdm(file_list):
        seg_path = os.path.join(data_path, "seg", uid + "_seg.csv")
        bbox_path = os.path.join(data_path, "bbox", uid + "_bbox.csv")
        uid = uid.strip()
        face_crop[uid] = get_bbox(bbox_path)
        seg = pd.read_csv(seg_path)

        for idx, gt in seg.iterrows():
            personid = gt["person_id"]
            start_frame = int(gt["start_frame"])
            end_frame = int(gt["end_frame"])
            seg_length = end_frame - start_frame + 1
            save_id = f"{uid}_{personid}_{start_frame}_{end_frame}"

            ##### for setting maximum frame size and minimum frame size
            if (
                (mode == "train" and minframe != None and seg_length < minframe)
                or (seg_length <= 1)
                or (personid == 0)
            ):
                continue
            elif maxframe != None and seg_length > maxframe:
                it = int(seg_length / maxframe)
                for i in range(it):
                    sub_start = start_frame + i * maxframe
                    sub_end = min(end_frame, sub_start + maxframe)
                    sub_length = sub_end - sub_start + 1
                    if minframe != None and sub_length < minframe:
                        continue
                    segments.append([uid, personid, sub_start, sub_end, save_id])
            else:
                segments.append([uid, personid, start_frame, end_frame, save_id])

    return segments, face_crop


class test_ImagerLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        video_path,
        maxframe,
        minframe,
        mode="test",
        transform=None,
        img_size=128,
    ):
        self.audio_path = "./extracted_audio"
        self.video_path = video_path
        self.file_list = testmakeFileList(data_path)
        print(f"{mode} file with length: {str(len(self.file_list))}")

        self.mode = mode
        segments, face_crop = test_make_dataset(
            self.file_list, data_path, maxframe, minframe, self.mode
        )
        print("finish making dataset")
        self.segments = segments
        self.face_crop = face_crop
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, indices):
        source_video = self._get_video(indices)
        source_audio = self._get_audio(indices)
        sid = self.segments[indices][4]
        return source_video, source_audio, sid

    def __len__(self):
        return len(self.segments)

    def _get_video(self, index, debug=False):
        video_size = self.img_size
        uid, personid, start_frame, end_frame, _ = self.segments[index]
        cap = cv2.VideoCapture(os.path.join(self.video_path, f"{uid}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        video = []
        for i in range(start_frame, end_frame + 1):
            key = str(i) + ":" + str(personid)
            # print("key: ", key)
            # print("face: ", dict(list(self.face_crop[uid].items())[:3]))
            if key in self.face_crop[uid].keys():
                bbox = self.face_crop[uid][key]
                if os.path.isfile(
                    f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png"
                ):
                    img = cv2.imread(
                        f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png"
                    )
                    face = cv2.resize(img, (video_size, video_size))
                else:
                    ret, img = cap.read()

                    if not ret:
                        print("not ret")
                        video.append(
                            np.zeros((1, video_size, video_size, 3), dtype=np.uint8)
                        )
                        continue

                    if not os.path.isdir(f"./extracted_frames/{uid}"):
                        os.mkdir(f"./extracted_frames/{uid}")
                    x1, y1, x2, y2 = (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    )

                    face = img[y1:y2, x1:x2, :]
                    if face.size != 0:
                        print(f"{uid}/write: {i:05d}_{personid}")
                        cv2.imwrite(
                            f"./extracted_frames/{uid}/img_{i:05d}_{personid}.png", face
                        )
                        logger.info(f"writing {uid}/img_{i:05d}_{personid}.png")
                        
                try:
                    face = cv2.resize(face, (video_size, video_size))
                except:
                    # bad bbox
                    face = np.zeros((video_size, video_size, 3), dtype=np.uint8)

                if debug:
                    import matplotlib.pyplot as plt

                    plt.imshow(face)
                    plt.show()

                video.append(np.expand_dims(face, axis=0))
            else:
                print("not in face crop")
                video.append(np.zeros((1, video_size, video_size, 3), dtype=np.uint8))
                continue
        cap.release()
        video = np.concatenate(video, axis=0)
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)
        print(f"[get video] {uid}: ", video.shape)
        return video

    def _get_audio(self, index):
        uid, _, start_frame, end_frame, _ = self.segments[index]
        if not os.path.isfile(os.path.join(self.audio_path, f"{uid}.wav")):
            video = VideoFileClip(os.path.join(self.video_path, f"{uid}.mp4"))
            audio = video.audio
            audio.write_audiofile(os.path.join(self.audio_path, f"{uid}.wav"))

        audio, sample_rate = torchaudio.load(
            f"{self.audio_path}/{uid}.wav", normalize=True
        )

        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = transform(audio)
        # transform = torchaudio.transforms.DownmixMono(channels_first=True)
        # audio = transform(audio)
        audio = torch.mean(audio, dim=0)

        # onset = int(start_frame / 30 * 16000)
        # offset = int(end_frame / 30 * 16000)
        onset = int(max((start_frame - 30) / 30 * 16000, 0))
        offset = int(min((end_frame + 30) / 30 * 16000, 4799999))
   
        crop_audio = audio[onset:offset]
        if crop_audio.size()[0] == 0:
            logger.info(f"{uid} size is equal to 0, resaving")
            video = VideoFileClip(os.path.join(self.video_path, f"{uid}.mp4"))
            audio = video.audio
            audio.write_audiofile(os.path.join(self.audio_path, f"{uid}.wav"))
            logger.info(f"finish resaving!")
        return crop_audio.to(torch.float32)

    def _get_target(self, index):
        if self.mode == "train":
            return torch.LongTensor([self.segments[index][2]])
        else:
            return self.segments[index]
