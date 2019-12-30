# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import *
from config import *


def add_watermark(cap, out=None):
    for frame_num, frame in frames(cap):
        cv2.imshow('frame', frame)
        print (frame.shape)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        rows, cols, _ = yuv.shape
        Y = yuv[:, :, 0]
        print ("Y: ", Y.shape)
        y = np.copy(Y)

        block_rows = rows//BLOCK
        block_cols = cols//BLOCK
        shape = (block_rows, block_cols, BLOCK, BLOCK)
        print (Y.itemsize, Y.strides)
        strides = 3 * Y.itemsize * np.array([BLOCK*cols, BLOCK, cols, 1])
        block_y = np.lib.stride_tricks.as_strided(Y, shape=shape, strides=strides)
        print (block_y.shape, block_y.size, block_y.dtype)

        for i in range(block_rows):
            for j in range(block_cols):
                block = block_y[i, j]
                avg = np.sum(np.reshape(block, (block.size, ))) / (BLOCK*BLOCK)

                sblock_rows = sblock_cols = BLOCK//SBLOCK
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
                # W[:,:,:,:] = 5.0

                w = None
                for ii in range(sblock_rows):
                    tmp = W[ii, 0]
                    for jj in range(1, sblock_cols):
                        tmp = np.concatenate([tmp, W[ii, jj]], 1)
                    w = np.concatenate([w, tmp]) if w is not None else tmp
                w *= WATERMARK[(frame_num//K) % WM_LEN]
                # print (w)

                block_float = block.astype(np.float64)
                # print (block[:8, :8])
                if frame_num % K < ((K-1)//2):
                    block_float += w
                elif frame_num % K > ((K-1)//2):
                    block_float -= w
                else:
                    pass
                block_float[block_float < 0] = 0
                block_float[block_float > 255] = 255
                # print (block_float)
                block_y[i, j] = block_float.astype(np.uint8)
                # print (block_y[i, j])
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

        # print (block_new[:8, :8])
        yuv[:, :, 0] = block_new
        # print (yuv)
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
        cv2.imshow('rgb', rgb)
        # print (cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)[:, :, 0])
        if cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)[:, :, 0].any() != yuv[:, :, 0].any():
            raise Exception

        # input()
        if out:
            out.write(rgb)


if __name__ == '__main__':
    path = "./color.avi"
    with open_video(path) as v:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('color_output.avi', fourcc, 30.0, (int(v.cap.get(3)), int(v.cap.get(4))))
        add_watermark(v.cap, out)
        out.release()

    cv2.destroyAllWindows()
