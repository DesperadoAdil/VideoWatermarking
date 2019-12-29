# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tqdm import tqdm

width = 576
height = 320

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('color.avi', fourcc, 30.0, (width, height))

for i in tqdm(range(900)):
    rgb = np.zeros((height, width, 3)).astype(np.uint8)

    rgb[:, :width//3, 2] = 255
    rgb[:, width//3:2*width//3, 1] = 255
    rgb[:, 2*width//3:width, 0] = 255

    cv2.imshow("frame", rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(rgb)

out.release()
