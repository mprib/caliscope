import pyxy3d.logger
import time
from queue import Queue
from threading import Thread, Event

import cv2
import numpy as np

import pyxy3d.calibration.draw_charuco as draw_charuco
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.interface import FramePacket, PointPacket
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.cameras.camera_array import CameraData

logger = pyxy3d.logger.get(__name__)

class IntrinsicCalibrator:
    """
    Takes a recorded stream and determines a CameraData object from it 
    Stream needs to have a charuco tracker assigned to it
    """ 
    def __init__(self, camera_data:CameraData):
        self.camera = camera_data  # reference needed to update params

        self.initialize_grid_history()


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
        self.all_obj_loc = []

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def add_corners(self, points:PointPacket):
        ids = points.point_id
        img_loc = points.img_loc
        obj_loc = points.obj_loc

        self.all_ids.append(ids)
        self.all_img_loc.append(img_loc)
        self.all_obj_loc.append(obj_loc)

        # self.update_grid_history(ids, img_loc)
        
    def update_grid_history(self, ids,img_loc):
        """
        Note that the connected points here comes from the charuco tracker.
        This grid history is likely best tracked by the controller and 
        a reference should be past to the frame emitter
        """
        if len(self.ids) > 2:
            self.grid_capture_history = draw_charuco.grid_history(
                self.grid_capture_history,
                ids,
                img_loc,
                self.connected_points,
            )

    def set_grid_frame(self):
        """Merges the current frame with the currently detected corners (red circles)
        and a history of the stored grid information."""


        self.grid_frame = self.frame_packet.frame_with_points
        self.grid_frame = cv2.addWeighted(
            self.grid_frame, 1, self.grid_capture_history, 1, 0
        )

        self.grid_frame_ready_q.put("frame ready")


    def calibrate(self):
        """
        Use the recorded image corner positions along with the objective
        corner positions based on the board definition to calculated
        the camera matrix and distortion parameters
        """
        logger.info(f"Calibrating camera {self.camera.port}....")

        objpoints = self.all_obj_loc
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
        logger.info(f"Grid Count: {self.grid_count}")

    def update_camera(self):
        logger.info(f"Setting calibration params on camera {self.camera.port}")

        # ret is RMSE of reprojection
        self.camera.error = round(self.error, 3)
        self.camera.matrix = self.mtx
        self.camera.distortions = self.dist
        self.camera.grid_count = self.grid_count

