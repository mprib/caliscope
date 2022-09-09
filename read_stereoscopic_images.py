import cv2 as cv

capture_0 = cv.VideoCapture("anipose_videos\calibration\calib-charuco-camA.MOV")
capture_1 = cv.VideoCapture("anipose_videos\calibration\calib-charuco-camB.MOV")

num = 0

while capture_0.isOpened():

    success_0, img_0 = capture_0.read()
    success_1, img_1 = capture_0.read()

    k = cv.waitKey(5)

    if k == 27: # ESC to stop   
        break
    elif k == ord('s'): # 
        cv.imwrite('images/imageL' + str(num) + '.png', img_0)
        cv.imwrite('images/imageR' + str(num) + '.png', img_1)
        num+=1

    cv.imshow("Img 0", img_0)
    cv.imshow("Img 1", img_1)