# -*- coding: utf-8 -*-
import cv2


class Frame(object):
    def __init__(self, index, diff):
        self.index = index
        self.diff = diff

    def __repr__(self):
        return '<index: %r, diff: %r>' % (self.index, self.diff)


class open_video(object):
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print ("Video Exit With: %s, %s" % (exc_type, exc_val))
        self.cap.release()


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


def get_frame(cap, index):
    if cap.isOpened():
        if hasattr(cv2, 'cv'):
            cap.set(cv2.cv.CAP_PROP_POS_FRAMES, index)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        return frame if ret else None
    return None
