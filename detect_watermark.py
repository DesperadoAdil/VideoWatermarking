# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scene_detect import scene_detect
from utils import *
from config import *


def detect_watermark(cap, scenes):
    for index, frame in frames(cap):
        cv2.imshow("frame", frame)


if __name__ == '__main__':
    path = "./test_output.avi"
    with open_video(path) as v:
        scenes = scene_detect(v.cap)
        print (scenes)

    with open_video(path) as v:
        detect_watermark(v.cap, scenes)

    cv2.destroyAllWindows()
