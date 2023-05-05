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
from pyxy3d.triangulate.sync_packet_triangulator import SyncPacketTriangulator
from pyxy3d.interface import FramePacket, TrackerFactory

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents

session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration_2_cam")
copy_session_path = Path(
    __root__, "tests", "sessions_copy_delete", "mediapipe_calibration_2_cam"
)
copy_contents(session_path, copy_session_path)


config = Configurator(copy_session_path)


def create_xy_points(
    config: Configurator, recording_directory: Path, tracker_factory: TrackerFactory
):
    camera_array: CameraArray = config.get_camera_array()
    ports = camera_array.cameras.keys()

    recording_directory = Path(copy_session_path, "calibration", "extrinsic")

    frame_times = pd.read_csv(Path(recording_directory, "frame_time_history.csv"))
    sync_index_count = len(frame_times["sync_index"].unique())

    stream_pool = RecordedStreamPool(
        directory=recording_directory,
        fps_target=100,
        tracker_factory=tracker_factory,
        config_path=config.toml_path,
    )
    synchronizer = Synchronizer(stream_pool.streams, fps_target=100)
    video_recorder = VideoRecorder(synchronizer)
    video_recorder.start_recording(
        destination_folder=recording_directory,
        include_video=True,
        show_points=True,
        suffix="_xy",
    )
    stream_pool.play_videos()

    while video_recorder.recording:
        sleep(1)
        percent_complete = int((video_recorder.sync_index / sync_index_count) * 100)
        logger.info(f"{percent_complete}% processed")
