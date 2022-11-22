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
from queue import Queue

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.calibration.charuco import Charuco
from src.calibration.corner_tracker import CornerTracker
from src.cameras.camera import Camera


class MonoCalibrator:
    def __init__(self, camera, dispatcher, corner_tracker):

        # need camera to know resolution and to assign calibration parameters
        # to camera....but can also just get calibration output from here
        # and then assign it camera outside of this.
        self.camera = camera
        self.corner_tracker = corner_tracker
        self.dispatcher = dispatcher

        self.frame_q = Queue()
        port = self.camera.port        
        self.dispatcher.

        self.corner_ids = []
        self.corner_loc = []
        self.board_FOR_corner = []

        self.last_calibration_time = time.time()  # need to initialize to *something*

    def collect_corners(self, board_threshold=0.7, wait_time=0.5):

        corner_count = len(self.charuco.board.chessboardCorners)
        min_points_to_process = int(corner_count * board_threshold)

        if self._frame_corner_ids.any():
            enough_corners = len(self._frame_corner_ids) > min_points_to_process
        else:
            enough_corners = False

        enough_time_from_last_cal = time.time() > self.last_calibration_time + wait_time

        if enough_corners and enough_time_from_last_cal:

            # store the corners and IDs
            self.corner_loc.append(self._frame_corners)
            self.corner_ids.append(self._frame_corner_ids)

            # store objective corner positions in a board frame of reference
            # board_FOR_corners = self.charuco.board.chessboardCorners[self._frame_corner_ids, :]
            self.board_FOR_corner.append(self.board_FOR_corners)
            #
            self.update_capture_history()
            self.last_calibration_time = time.time()

    def update_capture_history(self):
        """
        Given a frame and the location of the charuco board corners within in,
        draw a line connecting the outer bounds of the detected corners and add
        it in to the history of captrued frames. One frame will hold the whole
        history of the corners collected.
        """

        possible_pairs = {
            pair for pair in combinations(self._frame_corner_ids.squeeze().tolist(), 2)
        }
        connected_pairs = self.connected_corners.intersection(possible_pairs)

        # build dictionary of corner positions:
        observed_corners = {}
        for crnr_id, crnr in zip(
            self._frame_corner_ids.squeeze(), self._frame_corners.squeeze()
        ):
            observed_corners[crnr_id] = (round(crnr[0]), round(crnr[1]))

        # add them to the visual representation of the grid capture history
        for pair in connected_pairs:
            point_1 = observed_corners[pair[0]]
            point_2 = observed_corners[pair[1]]

            cv2.line(self._grid_capture_history, point_1, point_2, (255, 165, 0), 1)

    def calibrate(self):
        """
        Use the recorded image corner positions along with the objective
        corner positions based on the board definition to calculated
        the camera matrix and distortion parameters
        """
        logging.info(f"Calibrating camera {self.camera.port}....")

        # organize parameters for calibration function
        objpoints = self.board_FOR_corner
        imgpoints = self.corner_loc
        height = self.image_size[0]
        width = self.image_size[1]

        error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (width, height), None, None
        )

        self.is_calibrated = True

        # ret is RMSE of reprojection
        self.camera.error = round(error, 3)
        self.camera.camera_matrix = mtx
        self.camera.distortion = dist
        self.camera.grid_count = len(self.corner_ids)

        logging.info(f"Error: {error}")
        logging.info(f"Camera Matrix: {mtx}")
        logging.info(f"Distortion: {dist}")
        logging.info(f"Grid Count: {self.camera.grid_count}")


if __name__ == "__main__":

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
    )
    cam = Camera(0)

    print(f"Using Optimized Code?: {cv2.useOptimized()}")
    calib = MonoCalibrator(cam, charuco)
    last_calibration_time = time.time()

    print("About to enter main loop")
    while True:

        read_success, frame = cam.capture.read()
        calib.find_corners(frame)
        calib.collect_corners(wait_time=0.5)
        merged_frame = calib.merged_grid_history()

        cv2.imshow("Press 'q' to quit", merged_frame)
        key = cv2.waitKey(1)

        # end capture when enough grids collected
        if key == ord("q"):
            cam.capture.release()
            cv2.destroyAllWindows()
            break

    calib.calibrate()
    print(f"Error: {cam.error}")
    print(f"Camera Matrix: {cam.camera_matrix}")
    print(f"Distortion: {cam.distortion}")
    print(f"Grid Count: {cam.grid_count}")
