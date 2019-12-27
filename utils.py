# -*- coding: utf-8 -*-
import cv2


def frames(cap=None):
    if not cap:
        return
    frame_num = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_num += 1
            yield (frame_num, frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
