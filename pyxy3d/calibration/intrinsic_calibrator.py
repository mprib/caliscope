import pyxy3d.logger
import time
from queue import Queue
from threading import Thread, Event

import cv2
import numpy as np

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

    def __init__(self, camera_data: CameraData, stream: RecordedStream):
        self.camera = camera_data  # pointer needed to update params
        self.stream = stream
        self.initialize_point_history()

        self.frame_packet_in = Queue()
        self.stream.subscribe(self.frame_packet_in)
        self.harvest_frames()

    def harvest_frames(self):
        self.stop_event = Event()
        self.stop_event.clear()

        def harvest_worker():
            while True:
                frame_packet = self.frame_packet_in.get()
                self.add_frame_packet(frame_packet)

                if self.stop_event.is_set():
                    break

        self.harvest_thread = Thread(target=harvest_worker, args=[], daemon=True)
        self.harvest_thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    @property
    def grid_count(self):
        """How many sets of corners have been collected up to this point"""
        return len(self.calibration_frame_indices)

    @property
    def image_size(self):
        image_size = list(self.camera.size)
        image_size.reverse()  # for some reason...
        image_size.append(3)

        return image_size

    def initialize_point_history(self):

        # list of frame_indices that will go into the calibration
        self.calibration_frame_indices = []

        # dictionaries here are indexed by frame_index
        self.all_ids = {}
        self.all_img_loc = {}
        self.all_obj_loc = {}

    def add_frame_packet(self, frame_packet: FramePacket):
        """
        Point data from frame packet is stored, indexed by the frame index
        """
        index = frame_packet.frame_index

        if index != -1:  # indicates end of stream
            self.all_ids[index] = frame_packet.points.point_id
            self.all_img_loc[index] = frame_packet.points.img_loc
            self.all_obj_loc[index] = frame_packet.points.obj_loc

            self.active_frame_index = index

    def add_calibration_frame_indices(self, frame_index: int):
        self.calibration_frame_indices.append(frame_index)

    def clear_calibration_data(self):
        self.calibration_frame_indices = []
        self.set_calibration_inputs()
        
        

    def set_calibration_inputs(self):
        self.calibration_point_ids = []
        self.calibration_img_loc = []
        self.calibration_obj_loc = []

        for index in self.calibration_frame_indices:
            id_count = len(self.all_ids[index])
            if id_count > 3:  # I believe this is a requirement of opencv
                self.calibration_point_ids.append(self.all_ids[index])
                self.calibration_img_loc.append(self.all_img_loc[index])
                self.calibration_obj_loc.append(self.all_obj_loc[index])
            else:
                logger.info(f"Note that empty data stored in frame index {index}. This is not being used in the calibration")


    def calibrate_camera(self):
        """
        Use the recorded image corner positions along with the objective
        corner positions based on the board definition to calculated
        the camera matrix and distortion parameters
        """
        self.set_calibration_inputs()

        logger.info(f"Calibrating camera {self.camera.port}....")

        width = self.stream.size[0]
        height = self.stream.size[1]

        self.error, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.calibration_obj_loc,
            self.calibration_img_loc,
            (width, height),
            None,
            None,
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
