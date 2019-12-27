# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import *

SYN_SEQ = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
WATERMARK = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
BLOCK = 32 # block size
SBLOCK = 8 # secondary block size
WATSON_NUMBER = 0.649 # Watson 感知模型提供的建议值
B = 2 # beita
K = 5 # mkae 1bit message hide in K frame
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
watermark = [i if i is 1 else -1 for i in (SYN_SEQ + WATERMARK)]
WM_LEN = len(watermark)
print (watermark)

path = "./test.avi"
cap = cv2.VideoCapture(path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_output.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))


for frame_num, frame in frames(cap):
    cv2.imshow('frame', frame)
    print (frame.shape)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    rows, cols, _ = yuv.shape
    Y = yuv[:, :, 0]
    print ("Y: ", Y.shape)
    y = np.copy(Y)

    block_rows = int(rows/BLOCK)
    block_cols = int(cols/BLOCK)
    shape = (block_rows, block_cols, BLOCK, BLOCK)
    print (Y.itemsize, Y.strides)
    strides = 3 * Y.itemsize * np.array([BLOCK*cols, BLOCK, cols, 1])
    block_y = np.lib.stride_tricks.as_strided(Y, shape=shape, strides=strides)
    print (block_y.shape, block_y.size, block_y.dtype)

    for i in range(block_rows):
        for j in range(block_cols):
            block = block_y[i, j]
            avg = np.sum(np.reshape(block, (block.size, ))) / (BLOCK*BLOCK)

            sblock_rows = sblock_cols = int(BLOCK/SBLOCK)
            sshape = (sblock_rows, sblock_rows, SBLOCK, SBLOCK)
            sstrides = block.itemsize * np.array([SBLOCK*BLOCK, SBLOCK, BLOCK, 1])
            sblock_y = np.lib.stride_tricks.as_strided(block, shape=sshape, strides=sstrides)

            """
            Su Q, Niu Y, Liu X. image watermarking algorithm based on dc components im lementin • in s atial domain J alaon rarh of omr, 2012.
            DC = sum(sblock_y[ii, jj]) / SBLOCK
            """
            DC = np.zeros((sblock_rows, sblock_cols))
            for ii in range(sblock_rows):
                for jj in range(sblock_cols):
                    sum_matrix = np.sum(np.reshape(sblock_y[ii, jj], (sblock_y[ii, jj].size, )))
                    DC[ii, jj] = sum_matrix / SBLOCK
            avg_dc = np.sum(np.reshape(DC, (DC.size, ))) / (sblock_rows*sblock_cols)

            """
            c(0, 0, k) = DC
            c(0, 0) = avg_dc
            a = WATSON_NUMBER
            t(t, j, k) = T(i, j) * (c(0, 0, k) / c(0, 0)) ** a
            ==> t(i, j, k) = T(i, j) * (c(0, 0, k) / c(0, 0)) ** a
            ==> t(i, j, k) = T(i, j) * (DC / avg_dc) ** WATSON_NUMBER
            """
            t = np.zeros(sshape)
            W = np.zeros(sshape)
            for ii in range(sblock_rows):
                for jj in range(sblock_cols):
                    t[ii, jj] = T * (DC[ii, jj] / avg_dc) ** WATSON_NUMBER
                    W[ii, jj] = t[ii, jj] / SBLOCK
            # W[:,:,:,:] = 3.0

            w = None
            for ii in range(sblock_rows):
                tmp = W[ii, 0]
                for jj in range(1, sblock_cols):
                    tmp = np.concatenate([tmp, W[ii, jj]], 1)
                w = np.concatenate([w, tmp]) if w is not None else tmp
            w *= watermark[int((frame_num/K) % WM_LEN)]

            block_float = block.astype(np.float64)
            if frame_num % K < (K-1) / 2:
                block_float += w
            elif frame_num % K > (K-1) / 2:
                block_float -= w
            else:
                pass
            block_float[block_float < 0] = 0
            block_float[block_float > 255] = 255
            block_y[i, j] = block_float.astype(np.uint8)
            # break
        # break

    block_new = None
    for i in range(block_rows):
        tmp = block_y[i, 0]
        for j in range(1, block_cols):
            tmp = np.concatenate([tmp, block_y[i, j]], 1)
        block_new = np.concatenate([block_new, tmp]) if block_new is not None else tmp

    if (block_new == y).all():
        print ("------------------------------!")

    yuv[:, :, 0] = block_new
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
    cv2.imshow('rgb', rgb)

        # input()
    out.write(rgb)


out.release()
cap.release()
cv2.destroyAllWindows()
