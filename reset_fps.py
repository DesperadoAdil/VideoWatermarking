# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import *
from config import *


def reset_fps(path):
    with open_video(path) as v:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('color_record3_output.avi', fourcc, FPS, (int(v.cap.get(3)), int(v.cap.get(4))))

        for frame_num, frame in frames(v.cap):
            out.write(frame)

        out.release()


if __name__ == '__main__':
    reset_fps("./color_record3_output.mp4")
