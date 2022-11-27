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
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src.calibration.draw_charuco as draw_charuco
from src.calibration.charuco import Charuco
from src.calibration.corner_tracker import CornerTracker


class MonoCalibrator:
    def __init__(
        self, camera, synchronizer, corner_tracker, board_threshold=0.7, wait_time=0.5
    ):

        self.camera = camera  # reference needed to update params
        self.port = camera.port
        self.corner_tracker = corner_tracker
        self.wait_time = wait_time
        self.capture_corners = False  # start out not doing anything

        self.synchronizer = synchronizer
        self.bundle_ready_q = Queue()
        self.grid_frame_ready_q = Queue()
        self.synchronizer.subscribe(self.bundle_ready_q)

        # TODO...this is going deeper into the hierarchy than I would like
        # and may deserve a refactor
        self.connected_corners = self.corner_tracker.charuco.get_connected_corners()
        board_corner_count = len(self.corner_tracker.charuco.board.chessboardCorners)
        self.min_points_to_process = int(board_corner_count * board_threshold)

        self.initialize_grid_history()

        self.last_calibration_time = time.time()  # need to initialize to *something*
        self.collecting_corners = True
        self.thread = Thread(target=self.collect_corners, args=(), daemon=True)
        self.thread.start()

    @property
    def grid_count(self):
        """How many sets of corners have been collected up to this point"""
        return len(self.all_ids)

    def initialize_grid_history(self):
        self.image_size = list(self.camera.resolution)
        self.image_size.reverse()  # for some reason...
        self.image_size.append(3)
        self.grid_capture_history = np.zeros(self.image_size, dtype="uint8")

        # roll back collected corners to the beginning
        self.all_ids = []
        self.all_img_loc = []
        self.all_board_loc = []

    def collect_corners(self):
        """
        Input: opencv frame

        Side Effect 1: records corner ids, positions, and board positions provided
        that enough time has past since the last set was recorded

        Side Effect 2: updates the image
        #TODO #13 Split out the image update to its own method that returns a modified frame
        """
        logging.debug("Entering collect_corners thread loop")
        while True:
            frame_bundle_notice = self.bundle_ready_q.get()
            self.frame = self.synchronizer.current_bundle[self.port]["frame"]

            self.ids = np.array([])
            self.img_loc = np.array([])
            self.board_loc = np.array([])

            if self.capture_corners:

                (
                    self.ids,
                    self.img_loc,
                    self.board_loc,
                ) = self.corner_tracker.get_corners(self.frame)

                if self.ids.any():
                    enough_corners = len(self.ids) > self.min_points_to_process
                else:
                    enough_corners = False

                enough_time_from_last_cal = (
                    time.time() > self.last_calibration_time + self.wait_time
                )

                if enough_corners and enough_time_from_last_cal:

                    # store the corners and IDs
                    self.all_ids.append(self.ids)
                    self.all_img_loc.append(self.img_loc)
                    self.all_board_loc.append(self.board_loc)

                    self.last_calibration_time = time.time()
                    self.update_grid_history()

            self.set_grid_frame()

    def update_grid_history(self):
        if len(self.ids) > 2:
            self.grid_capture_history = draw_charuco.grid_history(
                self.grid_capture_history,
                self.ids,
                self.img_loc,
                self.connected_corners,
            )

    def set_grid_frame(self):
        logging.debug(f"Frame Size is {self.frame.shape}")
        logging.debug(f"camera resolution is {self.camera.resolution}")

        # check to see if the camera resolution changed from the last round
        if (
            self.frame.shape[0] == self.grid_capture_history.shape[0]
            and self.frame.shape[1] == self.grid_capture_history.shape[1]
        ):
            grid_frame = cv2.addWeighted(self.frame, 1, self.grid_capture_history, 1, 0)
            grid_frame = draw_charuco.corners(grid_frame, self.ids, self.img_loc)

            self.grid_frame = grid_frame
            self.grid_frame_ready_q.put("frame ready")

        else:
            logging.debug("Reinitializing Grid Capture History")
            self.initialize_grid_history()
            self.grid_frame = self.grid_capture_history
            self.grid_frame_ready_q.put("frame ready")

    def calibrate(self):
        """
        Use the recorded image corner positions along with the objective
        corner positions based on the board definition to calculated
        the camera matrix and distortion parameters
        """
        logging.info(f"Calibrating camera {self.camera.port}....")

        self.collecting_corners = False

        objpoints = self.all_board_loc
        imgpoints = self.all_img_loc
        height = self.image_size[0]
        width = self.image_size[1]

        self.error, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (width, height), None, None
        )

        self.update_camera()
        self.is_calibrated = True

        logging.info(f"Error: {self.error}")
        logging.info(f"Camera Matrix: {self.mtx}")
        logging.info(f"Distortion: {self.dist}")
        logging.info(f"Grid Count: {self.camera.grid_count}")

    def update_camera(self):
        logging.info(f"Setting calibration params on camera {self.camera.port}")
        # ret is RMSE of reprojection
        self.camera.error = round(self.error, 3)
        self.camera.camera_matrix = self.mtx
        self.camera.distortion = self.dist
        self.camera.grid_count = len(self.all_ids)


if __name__ == "__main__":

    from src.cameras.camera import Camera
    from src.cameras.synchronizer import Synchronizer
    from src.cameras.video_stream import VideoStream

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
    )

    trackr = CornerTracker(charuco)
    test_port = 0
    cam = Camera(0)
    streams = {cam.port: VideoStream(cam)}

    syncr = Synchronizer(streams, fps_target=10)

    monocal = MonoCalibrator(cam, syncr, trackr)

    print("About to enter main loop")
    while True:
        # read_success, frame = cam.capture.read()
        # logging.debug("Placing frame in queue to process")
        # monocal.frame_ready_q.put({test_port: frame})
        frame_ready = monocal.grid_frame_ready_q.get()
        logging.debug("Getting grid frame to display")
        frame = monocal.grid_frame

        cv2.imshow("Press 'q' to quit", frame)
        key = cv2.waitKey(1)

        # end capture when enough grids collected
        if key == ord("q"):
            cam.capture.release()
            cv2.destroyAllWindows()
            break

    monocal.calibrate()
    monocal.update_camera()

    print(f"Error: {cam.error}")
    print(f"Camera Matrix: {cam.camera_matrix}")
    print(f"Distortion: {cam.distortion}")
    print(f"Grid Count: {cam.grid_count}")
