from types import DynamicClassAttribute
import cv2 as cv
import numpy as np

photo = "Photos\IMG_20140526_224627656.jpg"
video =   "anipose_videos\calibration\calib-charuco-camA.MOV"


img = cv.imread(photo)

# create a blank image to draw on so you don't change the original image
# 'uint8' is the data type for an image, the zeros make it blank
 
# dimensons of blank image are (height, width, # of color channels)
blank_img = np.zeros((500,500, 3), dtype='uint8')


# 1. paint the entire image a certain color
#blank_img[200:300, 300:400] = 0,0,255

cv.circle(blank_img, (250,250), 30, (255,0,0), thickness=3)
cv.rectangle(blank_img, (10, 100), (blank_img.shape[1]//2, blank_img.shape[0]//2), (0,0,255), thickness=cv.FILLED)


# Where I currently am in the tutorial
# https://www.youtube.com/watch?v=oXlwWbU8l2o&t=935s


# cv.imshow("original", img)

cv.imshow("drawn on", blank_img)

cv.waitKey(0)