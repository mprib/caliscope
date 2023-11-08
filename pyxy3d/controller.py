from typing import Optional
from PySide6.QtCore import QObject

import shutil
from threading import Thread, Event
from time import sleep
from pathlib import Path
from datetime import datetime
from os.path import exists
import numpy as np
import toml
from dataclasses import asdict
import cv2
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import pyxy3d.logger
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.cameras.camera import Camera
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.configurator import Configurator
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.interface import Tracker
from pyxy3d.playback_frame_emitter import PlaybackFrameEmitter

logger = pyxy3d.logger.get(__name__)


class Controller(QObject):
    """
    Thin layer to integrate GUI and backend
    """

    def __init__(self, workspace_dir: Path):
        super().__init__()
        self.workspace = workspace_dir
        self.config = Configurator(self.workspace)

        # streams will be used to play back recorded video with tracked markers to select frames
        self.all_camera_data = self.config.get_all_camera_data()
        self.intrinsic_streams = {}
        self.frame_emitters = {}
        self.intrinsic_calibrators = {}
        self.charuco = self.config.get_charuco()
        self.charuco_tracker = CharucoTracker(self.charuco)
        self.load_intrinsic_streams()

    def load_intrinsic_streams(self):
        source_directory = Path(self.workspace, "calibration", "intrinsic")

        for port, camera_data in self.all_camera_data.items():
            # data storage convention defined here
            source_file = Path(source_directory, f"port_{port}.mp4")
            size = camera_data.size
            rotation_count = camera_data.rotation_count
            source_properties = read_video_properties(source_file)
            assert size == source_properties["size"]  # just to make sure
            self.intrinsic_streams[port] = RecordedStream(
                directory=source_directory,
                port=port,
                size=size,
                rotation_count=rotation_count,
                tracker=self.charuco_tracker,
            )
            logger.info(f"Loading recorded stream stored in {source_file}")

    def add_camera_from_source(
        self, intrinsic_mp4: Path = None, port: int = None
    ) -> int:
        """
        When adding source video to calibrate a camera, the function returns the camera index
        File will be transferred to workspace/calibration/intrinsic/port_{index}.mp4
        in keeping with project layout
        """
        if port is None:
            port = len(self.all_camera_data)

        # copy source over to standard workspace structure
        intrinsic_source_dir = Path(self.workspace, "calibration", "intrinsic")
        target_mp4_path = Path(intrinsic_source_dir, f"port_{port}.mp4")
        shutil.copy(intrinsic_mp4, target_mp4_path)

        video_properties = read_video_properties(target_mp4_path)
        size = video_properties["size"]

        new_cam_data = CameraData(
            port=port,
            size=size,
            original_intrinsic_source=str(intrinsic_mp4)
        )
        self.all_camera_data[port] = new_cam_data
        self.config.save_all_camera_data(self.all_camera_data)

    def set_current_tracker(self, tracker: Tracker = None):
        self.tracker = tracker

    def play_stream(self,port):
        logger.info(f"Begin playing stream at port {port}")
        self.intrinsic_streams[port].play_video()

    def pause_stream(self, port):
        logger.info(f"Pausing stream at port {port}")
        self.intrinsic_streams[port].pause()

    def unpause_stream(self, port):
        logger.info(f"Unpausing stream at port {port}")
        self.intrinsic_streams[port].unpause()

    def stream_jump_to(self, port, frame_index):
        logger.info(f"Jump to frame {frame_index} at port {port}")
        self.intrinsic_streams[port].jump_to(frame_index)


def read_video_properties(source_path: Path) -> dict:
    # Dictionary to hold video properties
    properties = {}

    # Open the video file
    video = cv2.VideoCapture(str(source_path))
    logger.info(f"Attempting to open video file: {source_path}")

    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Could not open the video file: {source_path}")

    # Extract video properties
    properties["frame_count"] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    properties["fps"] = video.get(cv2.CAP_PROP_FPS)
    properties["width"] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties["height"] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    properties["size"] = (properties["width"], properties["height"])

    # Release the video capture object
    video.release()

    return properties
