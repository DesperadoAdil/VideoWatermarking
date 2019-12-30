# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scene_detect import scene_detect
from utils import *
from config import *

FRAME_CODE = {}


class Code(object):
    def __init__(self, index, code):
        self.index = index
        self.code = code
        self.next = None
        self.checked = False


def equal(x):
    mean = np.mean(x)
    for item in x:
        if item != mean and abs(item-mean) < EQUAL_DIFF:
            return False
    return True


def to_str(x):
    # print (x)
    return "-".join([str(i).replace("-1", "0") for i in x])


def search(index, i, scene_y, rows, cols):
    # print ("index: %d, i: %d, len: %d" % (index, i, len(scene_y)))
    if FRAME_CODE.get(index+i):
        return FRAME_CODE.get(index+i)

    detect_area = [y[rows//4:rows*3//4, cols//4:cols*3//4] for y in scene_y[i:i+K]]
    avg_y = [np.mean(y) for y in detect_area]
    mid = (K-1)//2
    if np.sum(avg_y[:mid])/mid >= avg_y[mid] and np.sum(avg_y[mid+1:])/mid <= avg_y[mid] \
        and equal(avg_y[:mid]) and equal(avg_y[mid+1:]):
        code = 1
        # print ("frame: %d" % (index+i))
        # print ("c: %s\n" % str(avg_y))
    elif np.sum(avg_y[:mid])/mid <= avg_y[mid] and np.sum(avg_y[mid+1:])/mid >= avg_y[mid] \
        and equal(avg_y[:mid]) and equal(avg_y[mid+1:]):
        code = -1
        # print ("frame: %d" % (index+i))
        # print ("b: %s\n" % str(avg_y))
    elif (i+index) % 5 == 0:
        # print ("-----------------frame: %d" % (index+i))
        # print ("b: %s\n" % str(avg_y))
        return []
    else:
        return []

    FRAME_CODE[index+i] = Code(index+i, code)
    if i+K+K <= len(scene_y):
        FRAME_CODE[index+i].next = search(index, i+K, scene_y, rows, cols)

    return FRAME_CODE[index+i]


def detect_watermark(cap, scenes):
    width, height = int(cap.get(3)), int(cap.get(4))
    print (width, height)

    ans = {}
    index = 0
    scenes = [Frame(899, 1)]
    for item in scenes:
        scene_len = item.index - index + 1
        if scene_len < S:
            index = item.index + 1
            continue
        # print ("scene: %d" % index)

        scene_frame = [get_frame(cap, i) for i in range(index, item.index+1)]
        rows, cols, _ = scene_frame[0].shape
        # print ("rows: %s, cols: %s" % (rows, cols))
        scene_y = [cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0] for frame in scene_frame]
        for i in range(scene_len-K+1):
            search(index, i, scene_y, rows, cols)

        for key, value in FRAME_CODE.items():
            if value.checked:
                continue
            code = value
            seq = [code.code]
            while code.next:
                code = code.next
                code.checked = True
                seq.append(code.code)
            # print ("seq_len: %d" % len(seq))
            # print (seq)

            syn_index = []
            for i in range(len(seq)-K+1):
                if seq[i:i+K] == SYN_SEQ:
                    syn_index.append(i)
            # print (syn_index)

            for i, it in enumerate(syn_index[:-1]):
                watermark = to_str(seq[it+K:syn_index[i+1]])
                # print (watermark)
                if watermark in ans:
                    ans[watermark] += 1
                else:
                    ans[watermark] = 1

        index = item.index + 1

    print ("同步序列：", str(SYN_SEQ))
    if ans != {}:
        for key, value in sorted(ans.items(), key=lambda x: x[1], reverse=True):
            print ("检测到水印序列 %s 共 %d 次" % (str(key.split("-")), value))
    else:
        print ("未检测到水印序列！")


if __name__ == '__main__':
    path = "./color_output.avi"
    with open_video(path) as v:
        scenes = scene_detect(v.cap)
        print ("镜头：", str(scenes))

    with open_video(path) as v:
        detect_watermark(v.cap, scenes)

    cv2.destroyAllWindows()
