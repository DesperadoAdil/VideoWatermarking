# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import *
from config import *


def scene_detect(cap, window_size=WINDOW_SIZE):
    def max_frame(window_frame):
        max_diff = index = -1
        for i, frame in enumerate(window_frame):
            if max_diff < frame.diff:
                max_diff = frame.diff
                index = i
        return window_frame[index]

    list_frames = []
    sus_max_frame = []
    window_frame = []
    pre_hist = None
    for frame_num, frame in frames(cap):
        shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = hist * (1.0 / (shape[0]*shape[1]))
        diff = np.sum(np.abs(np.subtract(hist, pre_hist))) if pre_hist is not None else 0
        list_frames.append(Frame(frame_num, diff))
        pre_hist = hist

    for frame in list_frames:
        window_frame.append(frame)

        if len(window_frame) < window_size:
            continue

        max_diff_frame = max_frame(window_frame)
        max_diff_index = max_diff_frame.index

        if not sus_max_frame:
            sus_max_frame.append(max_diff_frame)
            continue
        last_max_frame = sus_max_frame[-1]

        if (max_diff_index - last_max_frame.index) < SHOT_MIN_LEN:
            start_index = window_frame[0].index
            if last_max_frame.diff < max_diff_frame.diff:
                sus_max_frame[-1] = max_diff_frame
                pop_count = max_diff_index - start_index + 1
            else:
                pop_count = window_size
            window_frame = window_frame[pop_count:]
            continue

        sum_start_index = last_max_frame.index + 1
        sum_end_index = max_diff_index - 1

        sum_diff = sum([item.diff for item in list_frames[sum_start_index:sum_end_index+1]])
        average_diff = sum_diff / (sum_end_index - sum_start_index + 1)
        if max_diff_frame.diff >= (JUDGE_RATE * average_diff):
            sus_max_frame.append(max_diff_frame)

        window_frame = []

    if sus_max_frame[-1].index < list_frames[-1].index:
        sus_max_frame.append(list_frames[-1])

    return (sus_max_frame)


if __name__ == '__main__':
    path = "./test_output.avi"
    cap = cv2.VideoCapture(path)
    print (scene_detect(cap))
    cap.release()
    cv2.destroyAllWindows()
