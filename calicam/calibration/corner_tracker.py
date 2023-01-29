# There may be a mixed functionality here...I'm not sure. Between the corner
# detector and the corner drawer...like, there will need to be something that
# accumulates a frame of corners to be drawn onto the displayed frame.

import logging

LOG_FILE = "log\corner_tracker.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
from pathlib import Path

import cv2
import numpy as np

import calicam.calibration.draw_charuco
from calicam.calibration.charuco import Charuco


class CornerTracker:
    def __init__(self, charuco):

        # need camera to know resolution and to assign calibration parameters
        # to camera
        self.charuco = charuco
        self.board = charuco.board
        self.dictionary = self.charuco.board.dictionary

        # for subpixel corner correction
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.conv_size = (11, 11)  # Don't make this too large.

    def get_corners(self, frame):
        """Will check for charuco corners in the frame, if it doesn't find any, 
        then it will look for corners in the mirror image of the frame"""

        self.ids = np.array([])
        self.img_loc = np.array([])
        self.frame = frame

        # invert the frame for detection if needed
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        if self.charuco.inverted:
            self.gray = ~self.gray  # invert

        self.find_corners_single_frame(mirror=False)
        # print(self._frame_corner_ids)
        if not self.ids.any():
            # print("Checking mirror image")
            self.gray = cv2.flip(self.gray, 1)
            self.find_corners_single_frame(mirror=True)

        return self.ids, self.img_loc, self.board_loc

    def find_corners_single_frame(self, mirror):

        # detect if aruco markers are present
        aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(
            self.gray, self.dictionary
        )

        frame_width = self.frame.shape[1]  # used for flipping mirrored corners back

        # correct the mirror frame before putting text on it if it's flipped
        if mirror:
            self.frame = cv2.flip(self.frame, 1)

        # if so, then interpolate to the Charuco Corners and return what you found
        if len(aruco_corners) > 3:
            (success, _img_loc, _ids,) = cv2.aruco.interpolateCornersCharuco(
                aruco_corners, aruco_ids, self.gray, self.board
            )

            # This occasionally errors out...
            # only offers possible refinement so if it fails, just move along
            try:
                _img_loc = cv2.cornerSubPix(
                    self.gray,
                    _img_loc,
                    self.conv_size,
                    (-1, -1),
                    self.criteria,
                )
            except:
                pass

            if success:
                # assign to tracker
                self.ids = _ids
                self.img_loc = _img_loc

                # flip coordinates if mirrored image fed in
                if mirror:
                    self.img_loc[:, :, 0] = frame_width - self.img_loc[:, :, 0]

    @property
    def board_loc(self):
        """Objective position of charuco corners in a board frame of reference"""
        if self.ids.any():
            return self.charuco.board.chessboardCorners[self.ids, :]
        else:
            return np.array([])


if __name__ == "__main__":

    from calicam.cameras.camera import Camera

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    cam = Camera(0)

    print(f"Using Optimized Code?: {cv2.useOptimized()}")
    trackr = CornerTracker(charuco)

    print("About to enter main loop")
    while True:

        read_success, frame = cam.capture.read()
        ids, locations, board_corners = trackr.get_corners(frame)
        drawn_frame = calicam.calibration.draw_charuco.corners(frame, locations)

        cv2.imshow("Press 'q' to quit", drawn_frame)
        key = cv2.waitKey(1)

        # end capture when enough grids collected
        if key == ord("q"):
            cam.capture.release()
            cv2.destroyAllWindows()
            break
