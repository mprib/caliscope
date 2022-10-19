# This module is about having some fun and just seeing if I can get something weird
# to work that I think might have an application to the fullscale process later on.
from pathlib import Path
import sys
import numpy as np

import cv2
from charuco import Charuco
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera


charuco = Charuco(4,5,11,8.5,aruco_scale = .75, square_size_overide=.0525, inverted=True)
cam = Camera(0)

while True:

    success, frame = cam.capture.read()
    
    frame = cv2.flip(frame, 1)

    # invert the frame for detection if needed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
    if charuco.inverted:
        gray = ~ gray # invert
 
    aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(gray, charuco.board.dictionary)
    
    if len(aruco_corners)>0: 
        (success, 
        charuco_corners, 
        charuco_corner_ids) = cv2.aruco.interpolateCornersCharuco(
                                                                aruco_corners,
                                                                aruco_ids,
                                                                gray,
                                                                charuco.board)


        if success:
            charuco_corner_ids.tolist()
            charuco_corners.tolist()

            for ID, coord in zip(charuco_corner_ids[:,0], charuco_corners[:,0]):
                coord = list(coord)
                print(f"{ID}: {round(coord[0])}, {round(coord[1])}")
                # cv2.circle(frame, (round(coord[0]), round(coord[1])), 5,(120,120,0), 3)
                cv2.putText(frame,str(ID), (round(coord[0]), round(coord[1])), cv2.FONT_HERSHEY_SIMPLEX, .5,(120,120,0), 3)


    cv2.imshow("Charuco Test", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        cv2.destroyAllWindows()
        break

