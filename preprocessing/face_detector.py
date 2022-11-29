import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch.utils.data import Dataset

import dlib
from tqdm import tqdm

class VideoFaceDetector(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @property
    @abstractmethod
    def _batch_size(self) -> int:
        pass

    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass

'''   @abstractmethod
    def _detect_landmarks(self, frames) -> List:
        pass'''


class FacenetDetector(VideoFaceDetector):

    def __init__(self, device="cuda:0") -> None:
        super().__init__()
        self.detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)

    def _detect_faces(self, frames) -> List:
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        return [b.tolist() if b is not None else None for b in batch_boxes]

    '''def _detect_landmarks(self, frames) -> List:
        face_detector = dlib.get_frontal_face_detector()
        predictor_path = 'src/preprocess/shape_predictor_81_face_landmarks.dat'
        face_predictor = dlib.shape_predictor(predictor_path)
        faces = face_detector(frame, 1)

        if len(faces)==0:
            tqdm.write('No faces in {}:{}'.format(cnt_frame,os.path.basename(org_path)))
            continue
        face_s_max=-1
        landmarks=[]

        *_, batch_landmarks = self.detector.detect(frames, landmarks=True)
        return [b.tolist() if b is not None else None for b in batch_landmarks]
'''
    @property
    def _batch_size(self):
        return 32


class VideoDataset(Dataset):

    def __init__(self, videos) -> None:
        super().__init__()
        self.videos = videos

    def __getitem__(self, index: int):
        video = self.videos[index]
        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
        frames = OrderedDict()
        for i in range(frames_num):
            capture.grab()
            success, frame = capture.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize(size=[s // 2 for s in frame.size])
            frames[i] = frame
        return video, list(frames.keys()), list(frames.values())

    def __len__(self) -> int:
        return len(self.videos)
