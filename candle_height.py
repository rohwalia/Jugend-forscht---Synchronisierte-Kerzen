import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import os
from matplotlib.path import Path
from scipy import fftpack

frame_rate = 30 #frames per second
cap = cv2.VideoCapture("radiant_anti_1.mp4")
count=0
value = []
height = []
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    """plt.imshow(frame)
    plt.ylim(840, 120)
    plt.xlim(110, 540)
    plt.show()"""
    picture_median = cv2.medianBlur(frame, 3)
    #picture_median_gray = cv2.cvtColor(picture_median, cv2.COLOR_RGB2GRAY)
    otsu_red = cv2.threshold(picture_median[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    otsu_green = cv2.threshold(picture_median[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    otsu_blue = cv2.threshold(picture_median[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    otsu = (otsu_red + otsu_green + otsu_blue) / 3
    #print(otsu)
    canny_picture = cv2.Canny(picture_median, otsu / 2, otsu, apertureSize=3, L2gradient=True)
    #print(canny_picture)
    """if i==6:
        plt.imshow(frame)
        plt.ylim(840, 120)
        plt.xlim(110, 540)
        plt.savefig("picture_canny.svg")
        plt.imshow(canny_picture)
        plt.ylim(840, 120)
        plt.xlim(110, 540)
        plt.savefig("canny.svg")"""

    flag = 0
    for i in range(len(canny_picture)):
        if flag == 0:
            for j in canny_picture[i]:
                    if j!=0:
                        h = 1280-i
                        flag = 1
                        break
        else:
            break
    if flag == 0:
        h = 0
    height.append(h)
    print(h)
    count += 1
cap.release()
cv2.destroyAllWindows()
t = np.arange(0, count/frame_rate, 1/frame_rate)
print(t)
print(len(t))
print(height)
print(len(height))
height = np.array(height)
data = np.column_stack((height, t[0:len(height)]))
output = pd.DataFrame(data, columns=["h", "t"])
with pd.ExcelWriter('rad_anti_1.xlsx') as writer:
    sheetname = "Series"
    output.to_excel(writer, sheet_name=sheetname, index=False)
plt.plot(t[0:len(height)], height)
plt.xlabel("Time (s)")
plt.ylabel("Height")
plt.show()