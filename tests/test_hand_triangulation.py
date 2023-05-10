import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
from pyxy3d import __root__
import pytest
import shutil
import cv2
from pathlib import Path
import time
from pyxy3d.trackers.hand_tracker import HandTracker
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.interface import PointPacket, FramePacket, SyncPacket
from pyxy3d.triangulate.sync_packet_triangulator import SyncPacketTriangulator
from pyxy3d.cameras.camera_array import CameraArray, CameraData, get_camera_array
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d.configurator import Configurator
from pyxy3d.helper import copy_contents
from pyxy3d.trackers.tracker_enum import Tracker

# TEST_SESSIONS = ["mediapipe_calibration"]


def test_hand_tracker():
    """
    Just a basic tset to make sure that the hand tracker is working with the triangulator
    Asserts True if it finishes just to fit in with testing.
    """
    original_session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration")
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        "mediapipe_calibration",
    )

    # clear previous test so as not to pollute current test results
    if session_path.exists() and session_path.is_dir():
        logger.info(f"Removing previously copied sessions at {session_path}")
        shutil.rmtree(session_path)

    logger.info(
        f"Copying over files from {original_session_path} to {session_path} for testing purposes"
    )
    copy_contents(original_session_path, session_path)

    config = Configurator(session_path)

    logger.info(f"Creating RecordedStreamPool")
    recording_directory = Path(session_path, "calibration", "extrinsic")

    stream_pool = RecordedStreamPool(
        recording_directory,
        config=config,
        tracker=Tracker.HAND,
        fps_target=100,
    )
    logger.info("Creating Synchronizer")
    syncr = Synchronizer(stream_pool.streams, fps_target=100)

    #### Basic code for interfacing with in-progress RealTimeTriangulator
    #### Just run off of saved point_data.csv for development/testing
    camera_array: CameraArray = config.get_camera_array()
    sync_packet_triangulator = SyncPacketTriangulator(
        camera_array, syncr, output_directory=session_path
    )
    stream_pool.play_videos()

    while sync_packet_triangulator.running:
        logger.info("Waiting for streams to finish playing")
        time.sleep(1)

    # only getting here if the things runs
    assert True


if __name__ == "__main__":
    test_hand_tracker()
