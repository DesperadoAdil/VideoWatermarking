# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scene_detect import scene_detect
from utils import *
from config import *


def detect_watermark(cap, scenes):
    width, height = int(cap.get(3)), int(cap.get(4))
    print (width, height)

    """for index, frame in frames(cap):
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break"""
    index = 0
    for item in scenes:
        scene_len = item.index - index + 1
        if scene_len < S:
            index = item.index + 1
            continue
        print ("scene: %d" % index)

        scene_frame = [get_frame(cap, i) for i in range(index, item.index+1)]
        rows, cols, _ = scene_frame[0].shape
        print ("rows: %s, cols: %s" % (rows, cols))
        scene_y = [cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0] for frame in scene_frame]
        for i in range(scene_len-K+1):
            # detect_area = [y[rows//4:rows*3//4, cols//4:cols*3//4] for y in scene_y[i:i+K]]
            detect_area = [y[96:128, 160:192] for y in scene_y[i:i+K]]
            # print (detect_area)
            # print (detect_area[0].shape)
            avg_y = [np.mean(y) for y in detect_area]
            # print (avg_y)
            if avg_y[0] > avg_y[2] and avg_y[1] > avg_y[2] and avg_y[3] < avg_y[2] and avg_y[4] < avg_y[2]:
                print ("frame: %d" % (index+i))
                print ("c: %s\n" % str(avg_y))


        index = item.index + 1


    """i = 299
    while i >= 0:
        frame = get_frame(cap, i)
        cv2.imshow("frame", frame)
        i -= 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break"""


if __name__ == '__main__':
    path = "./test_output.avi"
    with open_video(path) as v:
        scenes = scene_detect(v.cap)
        print (scenes)

    with open_video(path) as v:
        detect_watermark(v.cap, scenes)

    cv2.destroyAllWindows()
