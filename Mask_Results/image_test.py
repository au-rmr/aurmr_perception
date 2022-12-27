import cv2

import numpy as np

img = cv2.imread("mask1.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Window", img)
cv2.waitKey(0)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
    	if(img[i][j] == 90):
            print(img[i][j], i, j)
