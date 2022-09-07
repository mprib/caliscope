
import cv2 as cv
from pathlib import Path


######################################################################
# View image

# img = cv.imread("Photos\IMG_20140526_224627656.jpg")
# cv.imshow('picture', img)
# cv.waitKey(0)


#####################################################################
# # Read in Video from webcams
# import cv2 as cv

# capture_0 = cv.VideoCapture(0)    # this reads in the first webcam
# capture_1 = cv.VideoCapture(1)
# # capture_2 = cv.VideoCapture(2)

# while True:
#     isTrue, frame_0 = capture_0.read()
#     cv.imshow('Video0', frame_0)

#     isTrue, frame_1 = capture_1.read()
#     cv.imshow('Video1', frame_1)

#     # isTrue, frame = capture_1.read()
#     # cv.imshow('Video', frame)

#     if cv.waitKey(20) & 0xFF==ord('d'): # basically means: stop if d is pressed
#         break

    
# capture_0.release()

# cv.destroyAllWindows()


#######################################################################
# Read in Video from File
import cv2 as cv
capture = cv.VideoCapture(r"C:\Users\Mac Prible\repos\learn-opencv\anipose_videos\calibration\calib-charuco-camA.MOV")   

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video0', frame)

    if cv.waitKey(20) & 0xFF==ord('d'): # basically means: stop if d is pressed
        break
    
capture.release()
cv.destroyAllWindows()