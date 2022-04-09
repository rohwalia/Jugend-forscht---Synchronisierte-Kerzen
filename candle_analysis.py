import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import os
from matplotlib.path import Path
from scipy import fftpack

frame_rate = 240 #frames per second
cap = cv2.VideoCapture("4_short.mp4")
i=0
value = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    picture_median = cv2.medianBlur(frame, 3)
    picture_median_gray = cv2.cvtColor(picture_median, cv2.COLOR_RGB2GRAY)
    otsu = cv2.threshold(picture_median_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if i==6:
        plt.imshow(frame)
        plt.ylim(840, 120)
        plt.xlim(110, 540)
        plt.savefig("picture.svg")
        plt.imshow(otsu)
        plt.ylim(840, 120)
        plt.xlim(110, 540)
        plt.savefig("otsu.svg")

    """otsu_real = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    print(otsu_real)
    frame_real = cv2.bitwise_and(frame, otsu_real)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow("mask", otsu_real)
    cv2.waitKey(0)"""

    #print(otsu.flatten())
    frame_flat = frame.reshape(-1, frame.shape[-1])
    #print(frame_flat)
    #print(otsu.flatten())
    #print(frame_flat)
    otsu_flat = np.where(otsu.flatten() != 0)
    #print(otsu_flat)
    filter = frame_flat[otsu_flat]
    value.append(np.median(filter, axis=0))
    i += 1
cap.release()
cv2.destroyAllWindows()
value = np.array(value)
t = np.arange(0, i/frame_rate, 1/frame_rate)
print(value)
print(t)
data = np.column_stack((value, t[:len(value)]))
output = pd.DataFrame(data, columns=["R", "G", "B", "t"])
with pd.ExcelWriter('4_short.xlsx') as writer:
    sheetname = "Series"
    output.to_excel(writer, sheet_name=sheetname, index=False)
plt.plot(t[:len(value)], value[:,0])
plt.xlabel("Time (s)")
plt.ylabel("R")
plt.show()