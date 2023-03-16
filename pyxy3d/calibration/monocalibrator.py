
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)
# import logging
# logger.setLevel(logging.DEBUG)
import time
from queue import Queue
from threading import Thread, Event

import cv2
import numpy as np


import pyxy3d.calibration.draw_charuco as draw_charuco
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.calibration.corner_tracker import CornerTracker
from pyxy3d.cameras.data_packets import FramePacket
from pyxy3d.cameras.live_stream import LiveStream

class MonoCalibrator():

    def __init__(
        self, stream:LiveStream,  board_threshold=0.7, wait_time=0.5
    ):
        self.stream = stream
        self.camera: Camera = stream.camera  # reference needed to update params
        self.port = self.camera.port
        self.wait_time = wait_time
        self.capture_corners = Event()
        self.capture_corners.clear() # start out not doing anything
        self.stop_event = Event()
        
        self.frame_packet_in_q = Queue(-1)    
        self.subscribe_to_stream()

        self.grid_frame_ready_q = Queue()
        self.connected_corners = self.stream.charuco.get_connected_corners()

        board_corner_count = len(self.stream.charuco.board.chessboardCorners)
        self.min_points_to_process = int(board_corner_count * board_threshold)

        self.initialize_grid_history()

        self.last_calibration_time = (
            time.perf_counter()
        )  # need to initialize to *something*

        self.thread = Thread(target=self.collect_corners, args=(), daemon=True)
        self.thread.start()

        logger.info(f"Beginning monocalibrator for port {self.port}")

     
    @property
    def grid_count(self):
        """How many sets of corners have been collected up to this point"""
        return len(self.all_ids)

    @property
    def image_size(self):
        image_size = list(self.camera.size)
        image_size.reverse()  # for some reason...
        image_size.append(3)

        return image_size

    def initialize_grid_history(self):
        self.grid_capture_history = np.zeros(self.image_size, dtype="uint8")

        # roll back collected corners to the beginning
        self.all_ids = []
        self.all_img_loc = []
        self.all_board_loc = []
    
    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def subscribe_to_stream(self):
        self.stream.subscribe(self.frame_packet_in_q)
    
    def unsubscribe_to_stream(self): 
        self.stream.unsubscribe(self.frame_packet_in_q)
    

    def collect_corners(self):
        """
        Input: opencv frame

        Primary Action: records corner ids, positions, and board positions provided
        that enough time has past since the last set was recorded

        """
        logger.debug("Entering collect_corners thread loop")
        
        # self.stream.push_to_out_q.set()
        
        while not self.stop_event.is_set():
            
            self.frame_packet: FramePacket = self.frame_packet_in_q.get()
            self.frame = self.frame_packet.frame

            if self.capture_corners.is_set() and self.frame_packet.points is not None:
                logger.info("Points found and being processed...")
                self.ids = self.frame_packet.points.point_id
                self.img_loc = self.frame_packet.points.img_loc
                self.board_loc = self.frame_packet.points.board_loc

                if self.ids.any():
                    enough_corners = len(self.ids) > self.min_points_to_process
                else:
                    enough_corners = False

                enough_time_from_last_cal = (
                    time.perf_counter() > self.last_calibration_time + self.wait_time
                )

                if enough_corners and enough_time_from_last_cal:

                    # store the corners and IDs
                    self.all_ids.append(self.ids)
                    self.all_img_loc.append(self.img_loc)
                    self.all_board_loc.append(self.board_loc)

                    self.last_calibration_time = time.perf_counter()
                    self.update_grid_history()

            if self.frame_packet.frame is not None:
                self.set_grid_frame()

        logger.info(f"Monocalibrator at port {self.port} successfully shutdown...")
        self.stream.push_to_out_q.clear()
        
    def update_grid_history(self):
        if len(self.ids) > 2:
            self.grid_capture_history = draw_charuco.grid_history(
                self.grid_capture_history,
                self.ids,
                self.img_loc,
                self.connected_corners,
            )

    def set_grid_frame(self):
        """Merges the current frame with the currently detected corners (red circles) 
        and a history of the stored grid information."""

        logger.debug(f"Frame Size is {self.frame.shape} at port {self.port}")
        logger.debug(
            f"camera resolution is {self.camera.size} at port {self.port}"
        )

        # check to see if the camera resolution changed from the last round
        if (
            self.frame.shape[0] == self.grid_capture_history.shape[0]
            and self.frame.shape[1] == self.grid_capture_history.shape[1]
        ):
            self.frame_packet.frame = cv2.addWeighted(self.frame_packet.frame, 1, self.grid_capture_history, 1, 0)
            draw_charuco.corners(self.frame_packet)

            self.grid_frame = self.frame_packet.frame
            self.grid_frame_ready_q.put("frame ready")

        else:
            logger.debug("Reinitializing Grid Capture History")
            self.initialize_grid_history()
            self.grid_frame = self.grid_capture_history
            self.grid_frame_ready_q.put("frame ready")

    def calibrate(self):
        """
        Use the recorded image corner positions along with the objective
        corner positions based on the board definition to calculated
        the camera matrix and distortion parameters
        """
        logger.info(f"Calibrating camera {self.camera.port}....")

        objpoints = self.all_board_loc
        imgpoints = self.all_img_loc
        height = self.grid_capture_history.shape[0]
        width = self.grid_capture_history.shape[1]

        self.error, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (width, height), None, None
        )

        # fix extra dimension in return value of cv2.calibrateCamera
        self.dist = self.dist[0]

        self.update_camera()
        self.is_calibrated = True

        logger.info(f"Error: {self.error}")
        logger.info(f"Camera Matrix: {self.mtx}")
        logger.info(f"Distortion: {self.dist}")
        logger.info(f"Grid Count: {self.camera.grid_count}")

    def update_camera(self):
        logger.info(f"Setting calibration params on camera {self.camera.port}")
        # ret is RMSE of reprojection
        self.camera.error = round(self.error, 3)
        self.camera.matrix = self.mtx
        self.camera.distortions = self.dist
        self.camera.grid_count = len(self.all_ids)

if __name__ == "__main__":

    from pyxy3d.cameras.camera import Camera
    from pyxy3d.cameras.synchronizer import Synchronizer
    from pyxy3d.cameras.live_stream import LiveStream

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )

    test_port = 0
    cam = Camera(0)
    stream = LiveStream(cam, charuco=charuco)
    stream._show_fps = True
    # syncr = Synchronizer(streams, fps_target=20)

    monocal = MonoCalibrator(stream)

    monocal.capture_corners.set()
    
    print("About to enter main loop")
    while True:
        # read_success, frame = cam.capture.read()
        frame_ready = monocal.grid_frame_ready_q.get()
        logger.debug("Getting grid frame to display")
        frame = monocal.grid_frame

        cv2.imshow("Press 'q' to quit", frame)
        key = cv2.waitKey(1)

        # end capture when enough grids collected
        if key == ord("q"):
            cam.capture.release()
            cv2.destroyAllWindows()
            break

        if key == ord("t"):
            if monocal.stream.track_points.is_set():
                monocal.stream.track_points.clear()
            else:
                monocal.stream.track_points.set()
    
        if key == ord("v"):
            stream.change_resolution((1280,720))

    monocal.calibrate()
    monocal.update_camera()

    print(f"Error: {cam.error}")
    print(f"Camera Matrix: {cam.matrix}")
    print(f"Distortions: {cam.distortions}")
    print(f"Grid Count: {cam.grid_count}")
