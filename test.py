# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scene_detect import scene_detect
from utils import *
from config import *


def equal(x, y):
    return x == y or abs(x-y) < EQUAL_DIFF


def detect_watermark(cap):
    width, height = int(cap.get(3)), int(cap.get(4))
    print (width, height)

    """for index, frame in frames(cap):
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break"""
    index = 0
    scenes = [Frame(299, 1)]
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

            """with open_video("./tiny.avi") as v:
                print ("\npre:")
                scene_frame1 = [get_frame(v.cap, i) for i in range(index, item.index+1)]
                scene_y1 = [cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0] for frame in scene_frame1]
                detect_area1 = [y[:32, :32] for y in scene_y1[0:K]]
                avg_y1 = [np.mean(y) for y in detect_area1]
                print (avg_y1)
                print (detect_area1[1])"""

            # print ("\nnow:")
            # detect_area = [y[rows//4:rows*3//4, cols//4:cols*3//4] for y in scene_y[i:i+K]]
            detect_area = [y[:32, :32] for y in scene_y[i:i+K]]
            # print (detect_area)
            # print (detect_area[0].shape)
            avg_y = [np.mean(y) for y in detect_area]
            # print (avg_y)
            # print (detect_area[1])

            if (avg_y[0]+avg_y[1])/2 >= avg_y[2] and (avg_y[3]+avg_y[4])/2 <= avg_y[2] \
                and equal(avg_y[0], avg_y[1]) and equal(avg_y[3], avg_y[4]):
                print ("frame: %d" % (index+i))
                print ("c: %s\n" % str(avg_y))
            elif (avg_y[0]+avg_y[1])/2 <= avg_y[2] and (avg_y[3]+avg_y[4])/2 >= avg_y[2] \
                and equal(avg_y[0], avg_y[1]) and equal(avg_y[3], avg_y[4]):
                print ("frame: %d" % (index+i))
                print ("b: %s\n" % str(avg_y))
            elif i % 5 == 0:
                print ("-----------------frame: %d" % (index+i))
                print ("b: %s\n" % str(avg_y))


            """print ("\ndiff:")
            diff = detect_area[0] - detect_area1[0]
            print (diff)"""

            # if i == 5:
            # break


        index = item.index + 1


def test(cap):
    pre_cap = cv2.VideoCapture("./tiny.avi")

    for i in range(300):
    # for i, scene_frame in frames(cap):
        print ("frame: %d" % i)
        print ("\npre:")
        scene_frame1 = get_frame(pre_cap, i)
        scene_y1 = cv2.cvtColor(scene_frame1, cv2.COLOR_BGR2YCrCb)
        detect_area1 = scene_y1[:, :, 0]
        avg_y1 = np.mean(detect_area1)
        print (avg_y1)
        print (detect_area1.shape)
        print (detect_area1[:8, :8])

        print ("\nnow:")
        scene_frame = get_frame(cap, i)
        detect_area = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        avg_y = np.mean(detect_area)
        print (avg_y)
        print (detect_area.shape)
        print (detect_area[:8, :8])

        input()


def diff(cap):
    last_y = cv2.cvtColor(get_frame(cap, 0), cv2.COLOR_BGR2YCrCb)[:, :, 0]
    for i in range(1, 300):
        print ("\nframe %d - frame %d" % (i, i-1))
        scene_frame = get_frame(cap, i)
        y = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        print ("avg_y: %.6f" % np.mean(y))
        # print (y.shape)
        # print (y)
        diff = y - last_y
        print (diff[:8, :8])

        input()

        last_y = y


if __name__ == '__main__':

    path = "./tiny_output.avi"
    with open_video(path) as v:
        # detect_watermark(v.cap)
        # test(v.cap)
        diff(v.cap)

    cv2.destroyAllWindows()
