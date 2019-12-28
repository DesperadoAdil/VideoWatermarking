# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import *
from config import *


def detect_watermark(cap):
    for index, frame in frames(cap):
        cv2.imshow("frame", frame)


if __name__ == '__main__':
    path = "./test_output.avi"
    cap = cv2.VideoCapture(path)
    detect_watermark(cap)
    cap.release()
    cv2.destroyAllWindows()
