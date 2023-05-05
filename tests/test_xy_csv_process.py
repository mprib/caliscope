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

#%%
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
import pandas as pd
from pyxy3d.trackers.holistic_tracker import HolisticTrackerFactory
from pyxy3d.trackers.hand_tracker import HandTrackerFactory
from pyxy3d.trackers.pose_tracker import PoseTracker
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.recording.recorded_stream import RecordedStream, RecordedStreamPool
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.recording.video_recorder import VideoRecorder
from pyxy3d.triangulate.real_time_triangulator import SyncPacketTriangulator
from pyxy3d.interface import FramePacket

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents

session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration_2_cam")
copy_session_path = Path(
    __root__, "tests", "sessions_copy_delete", "mediapipe_calibration_2_cam"
)
copy_contents(session_path, copy_session_path)

config = Configurator(copy_session_path)
camera_array: CameraArray = config.get_camera_array()
ports = camera_array.cameras.keys()

# create a tracker
tracker_factory = HandTrackerFactory()
# tracker = PoseTracker()

recording_folder_path = Path(copy_session_path, "calibration", "extrinsic")

frame_times = pd.read_csv(Path(recording_folder_path, "frame_time_history.csv"))
max_sync_index = frame_times["sync_index"].max()

stream_pool = RecordedStreamPool(
    directory=recording_folder_path,
    fps_target=100,
    tracker_factory=tracker_factory,
    config_path=config.toml_path,
)
synchronizer = Synchronizer(stream_pool.streams, fps_target=100)
video_recorder = VideoRecorder(synchronizer)
video_recorder.start_recording(
    destination_folder=recording_folder_path,
    include_video=True,
    show_points=True,
    suffix="_xy",
)
stream_pool.play_videos()

processing_time = 0
while video_recorder.recording:
    sleep(1)
    processing_time += 1
    logger.info(f"Processing video data... {processing_time} seconds elapsed.")
    percent_complete = round((video_recorder.sync_index/max_sync_index)*100,0)
    logger.info(f"{percent_complete} % processed")

# make some basic assertions against the created file
produced_files = [
    Path(recording_folder_path, "xy.csv"),
    Path(recording_folder_path, "port_0_xy.mp4"),
    Path(recording_folder_path, "port_1_xy.mp4"),
]
for file in produced_files:
    logger.info(f"Asserting that the following file exists: {file}")
    assert file.exists()


# confirm that xy data is produced for the sync indices (slightly reduced to avoid missing data issues)
xy_data = pd.read_csv(Path(recording_folder_path, "xy.csv"))

logger.info(f"Max sync index: {max_sync_index} in frame history")
logger.info(f"Max sync index: {xy_data['sync_index'].max()} in xy.csv")
assert(max_sync_index == xy_data["sync_index"].max())

#%%