# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tqdm import tqdm

BLOCK = 32 # block size
SBLOCK = 8 # secondary block size
WATSON_NUMBER = 0.649 # Watson 感知模型提供的建议值
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

path = "./test.avi"
cap = cv2.VideoCapture(path)
frame_num = 0 # 300 frame

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('test1.avi', fourcc, 30.0, (576, 320))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_num += 1
        # cv2.imshow('frame',frame)
        # frame = cv2.resize(frame, (576, 320), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('frame',frame)
        print (frame.shape)
        # x = input()
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # print (yuv)
        print (yuv.shape, yuv.size, yuv.dtype)
        #yuv[1, 1, 0] = 10
        #print (yuv[1, 1])
        rows, cols, _ = yuv.shape
        Y = yuv[:, :, 0]
        print (Y.shape)
        print ("itemsize: %d" % Y.itemsize)

        block_rows = int(rows/BLOCK)
        block_cols = int(cols/BLOCK)
        # print (block_rows, block_cols)
        shape = (block_rows, block_cols, BLOCK, BLOCK)
        print (shape)
        strides = Y.itemsize * np.array([block_rows*cols, BLOCK, BLOCK*BLOCK, 1])
        print (strides)
        block_y = np.lib.stride_tricks.as_strided(Y, shape=shape, strides=strides)
        # print (block_y)
        print (block_y.shape, block_y.size, block_y.dtype)

        for i in range(block_rows):
            for j in range(block_cols):
                block = block_y[i, j]

                sblock_rows = sblock_cols = int(BLOCK/SBLOCK)
                print ("sbr: %d" % sblock_rows)
                sshape = (sblock_rows, sblock_rows, SBLOCK, SBLOCK)
                print (sshape)
                sstrides = block.itemsize * np.array([sblock_rows*BLOCK, SBLOCK, SBLOCK*SBLOCK, 1])
                print (sstrides)
                sblock_y = np.lib.stride_tricks.as_strided(block, shape=sshape, strides=sstrides)
                print (sblock_y.shape, sblock_y.size, sblock_y.dtype)

                avg = np.sum(np.reshape(block, (block.size, ))) / (BLOCK*BLOCK)
                print ("avg: %.3f" % avg)

                """
                Su Q, Niu Y, Liu X. image watermarking algorithm based on dc components im lementin • in s atial domain J alaon rarh of omr, 2012.
                DC = sum(sblock_y[ii, jj]) / SBLOCK
                """
                DC = np.zeros((sblock_rows, sblock_cols))
                for ii in range(sblock_rows):
                    for jj in range(sblock_cols):
                        # print (sblock_y[ii, jj])
                        sum_matrix = np.sum(np.reshape(sblock_y[ii, jj], (sblock_y[ii, jj].size, )))
                        print (sum_matrix)
                        DC[ii, jj] = sum_matrix / SBLOCK
                        # print ("DC: %.3f" % DC)
                print (DC)
                avg_dc = np.sum(np.reshape(DC, (DC.size, ))) / (sblock_rows*sblock_cols)
                print ("avg_dc: %.3f" % avg_dc)

                """
                c(0, 0, k) = DC
                c(0, 0) = avg_dc
                a = WATSON_NUMBER
                t(t, j, k) = T(i, j) * (c(0, 0, k) / c(0, 0)) ** a
                ==> t(0, 0, k) = T(0, 0) * (c(0, 0, k) / c(0, 0)) ** a
                ==> t(0, 0, k) = T(0, 0) * (DC / avg_dc) ** WATSON_NUMBER
                """
                t = T[0, 0] * (DC / avg_dc) ** WATSON_NUMBER
                print ("\nt[0, 0, k]:")
                print (t)

                # 修改
                pass

                break
            break

        input()
        # frame = cv2.flip(frame, -1)
        # out.write(frame)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print (frame_num)
# out.release()
cap.release()
cv2.destroyAllWindows()
