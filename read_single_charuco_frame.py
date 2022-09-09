# I have copied this from here:https://answers.opencv.org/question/98447/camera-calibration-using-charuco-and-python/
# looking for a starting point to begin to perform a charuco calibration.
# %%
from decimal import DecimalTuple
import cv2 as cv
import numpy as np

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# arguments: columns, rows, white space board?
board = cv.aruco.CharucoBoard_create(5,7,.025,.0125,dictionary)

frame = cv.imread("find_charuco.png")

allCorners = []
allIds = []

gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# are there any individual aruco markers detected?
corners, ids, rejected = cv.aruco.detectMarkers(gray,dictionary)


for c, id, rej in zip(corners, ids, rejected):
    print(f"Corner: {c} \nID: {id} \nRejected: {rej}")

# if so, then process the image
if len(corners)>0:
    # estimate where the charuco corners are given the identified arucos and board definition
    num_board_corners, board_corners, corner_id = cv.aruco.interpolateCornersCharuco(corners,ids,gray,board)

    if board_corners is not None and corner_id is not None and len(board_corners)>3:
        allCorners.append(board_corners)
        allIds.append(corner_id)
        
    # cv.aruco.drawDetectedMarkers(gray,corners,ids)

    for c, id in zip(board_corners, corner_id):
        cv.circle(gray, (round(c[0][0]), round(c[0][1])), 1,(255,0,0), thickness=-1)
        cv.putText(gray, str(id[0]), (round(c[0][0]), round(c[0][1])), cv.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1)
    
    # cv.aruco.drawDetectedMarkers(gray,board_corners,corner_id)

    cv.imshow("With markers", gray)

    cv.waitKey(0)


