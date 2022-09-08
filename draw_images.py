from types import DynamicClassAttribute
import cv2 as cv
import numpy as np
from resize_rescale import rescaleFrame


photo = "Photos\IMG_20140526_224627656.jpg"
video =   "anipose_videos\calibration\calib-charuco-camA.MOV"



def createShapes():

    img = cv.imread(photo)

    # create a blank image to draw on so you don't change the original image
    # 'uint8' is the data type for an image, the zeros make it blank
    

    # dimensons of blank image are (height, width, # of color channels)
    blank_img = np.zeros((500,500, 3), dtype='uint8')
    cv.imshow("blank", blank_img)
    
    cv.rectangle(blank_img, (10, 100), (blank_img.shape[1]//2, blank_img.shape[0]//2), (0,0,255), thickness=cv.FILLED)
    cv.imshow("rectangle", blank_img)

    cv.circle(blank_img, (250,250), 30, (255,0,0), thickness=3)
    cv.imshow("rectangle+circle", blank_img)


    line_start = (250,250)
    line_stop = (350, 350)

    cv.line(blank_img, line_start, line_stop, color = (255,255,255), thickness=1)
    cv.imshow("with line", blank_img)

    img = rescaleFrame(img, scale = 0.4)

    cv.putText(img, "Honeymoon!", (100,100), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 1)
    cv.imshow("Honeymoon", img)

    cv.waitKey(0)



if __name__ == "__main__":
    createShapes()


# 1. paint the entire image a certain color
#blank_img[200:300, 300:400] = 0,0,255



# Where I currently am in the tutorial
# https://www.youtube.com/watch?v=oXlwWbU8l2o&t=935s


# cv.imshow("original", img)
