# I have copied this from here:https://answers.opencv.org/question/98447/camera-calibration-using-charuco-and-python/
# looking for a starting point to begin to perform a charuco calibration.
# %%
from decimal import DecimalTuple
import cv2 as cv
import numpy as np

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# arguments: columns, rows, white space board?
board = cv.aruco.CharucoBoard_create(5,7,.025,.0125,dictionary)

capture = cv.VideoCapture(0)

allCorners = []
allIds = []
num = 0

while capture.isOpened():

    success, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    # are there any individual aruco markers detected?
    corners, ids, rejected = cv.aruco.detectMarkers(gray,dictionary)

    # if so, then process the image
    if len(corners)>0:
        # cv.aruco.drawDetectedMarkers(gray,corners,ids)
        # estimate where the charuco corners are given the identified arucos and board definition
        num_board_corners, board_corners, corner_id = cv.aruco.interpolateCornersCharuco(corners,ids,gray,board, 'MinMarkers', 1)

        if board_corners:
        # if board_corners is not None and corner_id is not None and len(board_corners)>3:
            allCorners.append(board_corners)
            allIds.append(corner_id)             
            for c, id in zip(board_corners, corner_id):
                cv.circle(gray, (round(c[0][0]), round(c[0][1])), 3,(0,255,255), thickness=-1)
                cv.putText(gray, str(id[0]), (round(c[0][0]), round(c[0][1])), cv.FONT_HERSHEY_PLAIN, 1.0, (255,0,255), 3)
        
    
    k = cv.waitKey(5)
    if k == 27: # ESC to stop   
        break
    elif k == ord('s'): 
        cv.imwrite('images/frame' + str(num) + '.png', frame)
        # cv.imwrite('images/imageR' + str(num) + '.png', img_1)
        num+=1

    cv.imshow("With markers", gray) 


print("All Corners:")
for crnr in allCorners:
    print(crnr)
    
print("All IDs")
for id in allIds:
    print(id)

capture.release()
cv.destroyAllWindows()


# %%
# Calibrate saved images