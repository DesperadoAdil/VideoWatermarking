# -*- coding: utf-8 -*-
import numpy as np

WINDOW_SIZE = 10
SHOT_MIN_LEN = 8
JUDGE_RATE = 6

SYN_SEQ = [1, 1, 1, 1, 1]
_WATERMARK = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1]
BLOCK = 32 # block size
SBLOCK = 8 # secondary block size
WATSON_NUMBER = 0.649 # Watson 感知模型提供的建议值
B = 2 # beita
K = 5 # mkae 1bit message hide in K frame, must be odd number
T = [
    [1.40, 1.01, 1.16, 1.66, 2.4, 3.43, 4.79, 6.56],
    [1.01, 1.45, 1.32, 1.52, 2.0, 2.71, 3.67, 4.93],
    [1.16, 1.32, 2.24, 2.59, 2.98, 3.64, 4.6, 5.88],
    [1.66, 1.52, 2.59, 3.77, 4.55, 5.3, 4.6, 7.6],
    [2.4, 2.00, 2.98, 4.55, 6.15, 7.46, 8.71, 10.17],
    [3.43, 2.71, 3.64, 5.3, 7.46, 9.62, 11.58, 13.51],
    [4.79, 3.67, 4.6, 6.28, 8.71, 11.58, 14.50, 17.29],
    [6.56, 4.93, 5.88, 7.6, 10.17, 13.51, 17.29, 21.15]
]
T = np.array(T)
WATERMARK = [i if i is 1 else -1 for i in (SYN_SEQ + _WATERMARK)]
print ("水印信息：", str(WATERMARK))
WM_LEN = len(WATERMARK)
SEQ_LEN = len(SYN_SEQ)
S = K * WM_LEN
EQUAL_DIFF = 5
SYN_SEQ_HAMMING_DISTANCE = 1
