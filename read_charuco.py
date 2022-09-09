# I have copied this from here:https://answers.opencv.org/question/98447/camera-calibration-using-charuco-and-python/
# looking for a starting point to begin to perform a charuco calibration.
# %%
from decimal import DecimalTuple
import time
import cv2 as cv
import numpy as np

# dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# # arguments: columns, rows, white space board?
# board = cv.aruco.CharucoBoard_create(5,7,.025,.0125,dictionary)
# img = board.draw((200*3,200*3))

# #Dump the calibration board to a file
# cv.imwrite('charuco.png',img)

# # %%
# #Start capturing images for calibration
# capture = cv.VideoCapture(0)

# allCorners = []
# allIds = []

# for decimator in range(0,500):
    
#     ret,frame = capture.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # are there any individual aruco markers detected?
#     corners, ids, rejected = cv.aruco.detectMarkers(gray,dictionary)

#     # if so, then process the image
#     if len(corners)>0:

#         # estimate where the charuco corners are given the identified arucos and board definition
#         num_board_corners, board_corner, corner_id = cv.aruco.interpolateCornersCharuco(corners,ids,gray,board)

#         if board_corner is not None and corner_id is not None and len(board_corner)>3 and decimator%3==0:
#             allCorners.append(board_corner)
#             allIds.append(corner_id)
            
#             # save out a still image to work with if type 'w'
#             # if cv.waitKey(20) & 0xFF == ord('w'):
#             #     print("Saving Image")
#             #     cv.imwrite("find_charuco.png",gray)
#             #     break

#         cv.aruco.drawDetectedMarkers(gray,corners,ids)



#     cv.putText(gray, f"Decimator: {decimator}", (100,100), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255,0), 1)

#     cv.imshow('frame',gray)
#     if cv.waitKey(20) & 0xFF == ord('q'):
#         break
    
#     decimator+=1

# imsize = gray.shape

# #Calibration fails for lots of reasons. Release the video if we do
# try:
#     cal = cv.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
# except:
#     capture.release()

# capture.release()
# cv.destroyAllWindows()
# # %%


dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# arguments: columns, rows, white space board?
board = cv.aruco.CharucoBoard_create(5,7,.025,.0125,dictionary)

capture = cv.VideoCapture(0)

allCorners = []
allIds = []

while True:

    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # are there any individual aruco markers detected?
    corners, ids, rejected = cv.aruco.detectMarkers(gray,dictionary)


    # for c, id, rej in zip(corners, ids, rejected):
    #     print(f"Corner: {c} \nID: {id} \nRejected: {rej}")

    # if so, then process the image
    if len(corners)>0:
        # estimate where the charuco corners are given the identified arucos and board definition
        num_board_corners, board_corners, corner_id = cv.aruco.interpolateCornersCharuco(corners,ids,gray,board)

        if board_corners is not None and corner_id is not None and len(board_corners)>3:
            allCorners.append(board_corners)
            allIds.append(corner_id)
            
            for c, id in zip(board_corners, corner_id):
                cv.circle(gray, (round(c[0][0]), round(c[0][1])), 1,(255,0,0), thickness=-1)
                cv.putText(gray, str(id[0]), (round(c[0][0]), round(c[0][1])), cv.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1)
            
    cv.imshow("Charuco", gray)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

imsize = gray.shape

#Calibration fails for lots of reasons. Release the video if we do
try:
    print("Calibrating....")
    cal = cv.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
    print(cal)
except:
    capture.release()

capture.release()
cv.destroyAllWindows()
