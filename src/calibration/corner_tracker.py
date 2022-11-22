# There may be a mixed functionality here...I'm not sure. Between the corner
# detector and the corner drawer...like, there will need to be something that
# accumulates a frame of corners to be drawn onto the displayed frame.

import logging

LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FILE = "monocalibrator.log"
logging.basicConfig(filename=LOG_FILE, filemode="w", level=LOG_LEVEL)

import sys
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.calibration.charuco import Charuco
from src.cameras.camera import Camera


class CornerTracker:
    def __init__(self, charuco):

        # need camera to know resolution and to assign calibration parameters
        # to camera
        self.charuco = charuco

        # for subpixel corner correction
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.conv_size = (11, 11)  # Don't make this too large.

    def get_corners(self, frame):
        """Will check for corners in the default board image, if it doesn't
        find any, then it will look for images in the mirror image of the
        default board"""

        self.corner_ids = np.array([])
        self.corner_loc = np.array([])
        self.frame = frame

        # invert the frame for detection if needed
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        if self.charuco.inverted:
            self.gray = ~self.gray  # invert

        self.find_corners_single_frame(mirror=False)
        # print(self._frame_corner_ids)
        if not self.corner_ids.any():
            # print("Checking mirror image")
            self.gray = cv2.flip(self.gray, 1)
            self.find_corners_single_frame(mirror=True)

        return self.corner_ids, self.corner_loc, self.board_FOR_corner

    def find_corners_single_frame(self, mirror):

        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(
            self.gray, self.charuco.board.dictionary
        )

        frame_width = frame.shape[1]  # used for flipping mirrored corners back

        # correct the mirror frame before putting text on it if it's flipped
        if mirror:
            self.frame = cv2.flip(self.frame, 1)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 3:
            (
                success,
                self.corner_loc,
                self.corner_ids,
            ) = cv2.aruco.interpolateCornersCharuco(
                aruco_corners, aruco_ids, self.gray, self.charuco.board
            )

            # This occasionally errors out...
            # only offers possible refinement so if it fails, just move along
            try:
                self.corner_loc = cv2.cornerSubPix(
                    self.gray,
                    self.corner_loc,
                    self.conv_size,
                    (-1, -1),
                    self.criteria,
                )
            except:
                pass

            if success:
                # clean up the data types
                self.corner_ids.tolist()
                self.corner_loc.tolist()

                # flip coordinates if mirrored image fed in
                if mirror:
                    self.corner_loc[:, :, 0] = frame_width - self.corner_loc[:, :, 0]

        #     else:
        #         self.corner_ids = np.array([])
        #         self.corner_loc = np.array([])
        # else:
        #     self.corner_ids = np.array([])
        #     self.corner_loc = np.array([])

    @property
    def board_FOR_corner(self):
        """Objective position of charuco corners in a board frame of reference"""
        if self.corner_ids.any():
            return self.charuco.board.chessboardCorners[self.corner_ids, :]
        else:
            return np.array([])


def draw_corners(frame, ids, locs):
    # TODO: break out into seperate method.... this is about drawing
    if len(ids) > 0:
        for _id, coord in zip(ids[:, 0], locs[:, 0]):
            coord = list(coord)
            # print(frame.shape[1])
            x = round(coord[0])
            y = round(coord[1])

            cv2.circle(frame, (x, y), 5, (0, 0, 220), 3)
            # cv2.putText(self.frame,str(ID), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5,(220,0,0), 3)
    return frame


if __name__ == "__main__":

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
    )
    cam = Camera(0)

    print(f"Using Optimized Code?: {cv2.useOptimized()}")
    calib = CornerTracker(charuco)

    print("About to enter main loop")
    while True:

        read_success, frame = cam.capture.read()
        ids, locations, board_corners = calib.get_corners(frame)
        drawn_frame = draw_corners(frame, ids, locations)

        cv2.imshow("Press 'q' to quit", drawn_frame)
        key = cv2.waitKey(1)

        # end capture when enough grids collected
        if key == ord("q"):
            cam.capture.release()
            cv2.destroyAllWindows()
            break
