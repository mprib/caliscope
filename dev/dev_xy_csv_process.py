"""
This may actually end up as a test (please, let that be the case...).
I want to set up a pipeline that could be used by the session to run the 
tracker point detection on each of the video files and save out
the xy point data. This may end up just being incapsulated in a simple
helper function that gets called from the session. I could do some
basic assertions at the end of it to make sure *something* happened.

All I need to do here is run the tracking on an mp4 and save the points out.
This is currenlty happening as part of the calibration process, but it should
also be happening during all of the initial processing.

This xy.csv file will provide the means to rapidly triangulate with 
filtering of the data using reprojection error...

"""

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)
from time import sleep
from queue import Queue
import cv2

import sys
from PyQt6.QtWidgets import QApplication
from pyxy3d.configurator import Configurator
from pathlib import Path
from pyxy3d import __root__
from pyxy3d.trackers.holistic_tracker import HolisticTrackerFactory
from pyxy3d.trackers.pose_tracker import PoseTracker
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.recording.recorded_stream import RecordedStream, RecordedStreamPool
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.recording.video_recorder import VideoRecorder
from pyxy3d.triangulate.real_time_triangulator import SyncPacketTriangulator
from pyxy3d.interface import FramePacket

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents

session_path = Path(__root__, "dev", "sample_sessions", "xy_points")
copy_session_path = Path(__root__, "dev", "sessions_copy_delete", "xy_points")
copy_contents(session_path, copy_session_path)


config = Configurator(copy_session_path)
camera_array: CameraArray = config.get_camera_array()
ports = camera_array.cameras.keys()

# create a tracker
tracker_factory = HolisticTrackerFactory()
# tracker = PoseTracker()

recording_folder_path = Path(copy_session_path, "recording_1")

stream_pool = RecordedStreamPool(directory=recording_folder_path,fps_target=100, tracker_factory=tracker_factory, config_path=config.toml_path)
synchronizer = Synchronizer(stream_pool.streams,fps_target=100)
video_recorder = VideoRecorder(synchronizer)
video_recorder.start_recording(destination_folder=recording_folder_path,include_video=True, show_points=True, suffix = "_xy")
stream_pool.play_videos()

while video_recorder.recording:
    sleep(1)