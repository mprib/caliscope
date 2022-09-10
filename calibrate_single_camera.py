# I have copied this from here:https://answers.opencv.org/question/98447/camera-calibration-using-charuco-and-python/
# looking for a starting point to begin to perform a charuco calibration.
# %%
from decimal import DecimalTuple
import cv2 as cv
import numpy as np


# %%
# create charuco for calibration
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# arguments: columns, rows, white space board?
charuco_border_inch = 0
charuco_height_inch = 11 # inches
charuco_width_inch = 8.5 # inches

paper_height_inch = charuco_height_inch + charuco_border_inch
paper_width_inch = charuco_width_inch + charuco_border_inch

# convert to meters
charuco_height = charuco_height_inch/39.37
charuco_width = charuco_width_inch/39.37

charuco_columns = 4
charuco_rows = 5
square_length = min([charuco_height/charuco_columns, 
                     charuco_height/charuco_rows, 
                     charuco_width/charuco_columns, 
                     charuco_width/charuco_rows]) 

aruco_length = square_length * 0.9 


board = cv.aruco.CharucoBoard_create(charuco_columns, charuco_rows, square_length, aruco_length, dictionary)

ppm = 300/39.37

cv.imwrite("charuco.png", ~board.draw((int(paper_width_inch*300), int(paper_height_inch*300))))

#############################################

# %%
capture = cv.VideoCapture(0)

board_corners = len(board.chessboardCorners)

corners_all = []
ids_all = []
image_size = None # determined at runtime

num = 0

while capture.isOpened():

    success, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    # If our image size is unknown, set it now
    if not image_size:
        image_size = gray.shape

    # are there any individual aruco markers detected?
    corners, ids, rejected = cv.aruco.detectMarkers(~gray,dictionary)

    # if so, then process the image
    if len(corners)>0:
        
        # cv.aruco.drawDetectedMarkers(gray,corners,ids)
    
        # estimate where the checkerboard corners are given the identified arucos and board definition
        num_board_corners, board_corners, corner_ids = cv.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            board, 
            minMarkers = 0)
        
        gray = cv.aruco.drawDetectedCornersCharuco(
                image=gray,
                charucoCorners=board_corners,
                charucoIds=corner_ids,
                cornerColor = (0,255,0))

        # draw checkerboard corners to visualize placement relative to arucos
        # if board_corners is not None and corner_ids is not None and len(board_corners)>3:

        #     for c, id in zip(board_corners, corner_ids):
        #         cv.circle(gray, (round(c[0][0]), round(c[0][1])), 3,(0,255,255), thickness=-1)
        #         cv.putText(gray, str(id[0]), (round(c[0][0]), round(c[0][1])), cv.FONT_HERSHEY_PLAIN, 1.0, (255,0,255), 1)
        
    
    k = cv.waitKey(5)
    if k == 27: # ESC to stop   
        break
    elif k == ord('s'): 
        cv.imwrite('images/frame' + str(num) + '.png', frame)
        # cv.imwrite('images/imageR' + str(num) + '.png', img_1)
        num+=1
    elif k == ord('c'):
        if board_corners is not None and corner_ids is not None and len(board_corners)>3:
            corners_all.append(board_corners)
            ids_all.append(corner_ids)    

    cv.imshow("With markers", gray) 


print("All Corners:")
for crnr in corners_all:
    print(crnr)
    
print("All IDs")
for id in ids_all:
    print(id)

capture.release()
cv.destroyAllWindows()


# %%
# Calibrate saved images
success, camera_matrix, distortion, rotation_vec, translation_vec = cv.aruco.calibrateCameraCharuco(
    charucoCorners=corners_all,
    charucoIds=ids_all,
    board=board,
    imageSize=image_size[::-1][0:2],
    cameraMatrix=None,
    distCoeffs=None)

# %%
