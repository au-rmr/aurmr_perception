from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str,
                    help='Path to a video or a sequence of image.', default='/home/soofiyanatar/datasets/Full_Dataset/rgb_all.png')
parser.add_argument('--algo', type=str,
                    help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

frame1 = cv.imread("/home/soofiyanatar/datasets/Full_Dataset/rgb_all.png")
frame = cv.imread("/home/soofiyanatar/datasets/Full_Dataset/rgb_empty.png")

frame = cv.resize(frame, (1080, 720), interpolation=cv.INTER_AREA)
frame1 = cv.resize(frame1, (1080, 720), interpolation=cv.INTER_AREA)

difference = cv.absdiff(frame1, frame)

# Threshold the difference image to get the initial mask
mask = difference.sum(axis=2) >= 100
mask = mask.astype(np.uint8)
mask = cv.bitwise_and(frame1, frame1, mask=mask)
print(mask)

# cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
# cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


cv.imshow('Frame', frame1)
cv.imshow('FG Mask', mask)

cv.waitKey(0)
