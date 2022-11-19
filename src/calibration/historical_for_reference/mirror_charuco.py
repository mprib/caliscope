# This module is about having some fun and just seeing if I can get something weird
# to work that I think might have an application to the fullscale process later on.

# Dang. That was kinda fun to be able to go from kludgey prototype to an 
# implemented feature in less than a day. And I think this will be clutch 
# when it comes to stereocalibration.



from pathlib import Path
import sys
import numpy as np

import cv2
from charuco import Charuco
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera


def mark_charuco_corners(frame, charuco, mirror):
    """Encapsulated function to mark frame"""

    # invert the frame for detection if needed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
    if charuco.inverted:
        gray = ~ gray # invert

    (aruco_corners,
     aruco_ids,
      rejected) = cv2.aruco.detectMarkers(gray, charuco.board.dictionary)
    
    frame_width = frame.shape[1]

    # correct the mirror frame before putting text on it if it's flipped
    if mirror:
        frame = cv2.flip(frame,1)

    if len(aruco_corners)>0: 
        (success, 
        charuco_corners, 
        charuco_corner_ids) = cv2.aruco.interpolateCornersCharuco(
                                                                aruco_corners,
                                                                aruco_ids,
                                                                gray, 
                                                                charuco.board)
        if success:
            # clean up the data types
            charuco_corner_ids.tolist()
            charuco_corners.tolist()


            for ID, coord in zip(charuco_corner_ids[:,0], charuco_corners[:,0]):
                coord = list(coord)
                # print(frame.shape[1])
                y = round(coord[1])
                # adjust horizontal position depending on mirroring
                if mirror:
                    x = round(frame_width - coord[0])
                else:
                    x = round(coord[0])

                cv2.circle(frame, (x, y), 5,(120,120,0), 3)
                cv2.putText(frame,str(ID), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5,(120,120,0), 3)

    return frame




# Test out functionality

# establish a connection with a camera
cam = Camera(0)

# Build a standard charuco...i think the mirror feature can be embedded in the
# calibrator
charuco = Charuco(4,5,11,8.5,aruco_scale = .75, square_size_overide=.0525, inverted=True)


while True:

    success, frame = cam.capture.read()


    # frame = cv2.flip(frame, 1)
    # check the frame for normal charuco and add if it's there
    frame = mark_charuco_corners(frame, charuco, mirror=False)
    # flip the frame and check for mirror charucos
    frame = cv2.flip(frame, 1)
    frame = mark_charuco_corners(frame, charuco, mirror=True)

    # flip the frame back
    cv2.imshow("Charuco Mirror Test", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        cv2.destroyAllWindows()
        break

