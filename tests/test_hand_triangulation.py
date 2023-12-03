from pyxy3d import __root__
import shutil
from pathlib import Path
import time
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.triangulate.sync_packet_triangulator import SyncPacketTriangulator
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.configurator import Configurator
from pyxy3d.helper import copy_contents
from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.recording.recorded_stream import RecordedStream
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

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

    logger.info("Creating RecordedStreamPool")
    recording_directory = Path(session_path, "calibration", "extrinsic")

    camera_array = config.get_camera_array()
    streams = {}
    for port, camera in camera_array.cameras.items():
        rotation_count = camera.rotation_count
        streams[port] = RecordedStream(
            recording_directory,
            port,
            rotation_count,
            fps_target=100,
            tracker=TrackerEnum.HAND.value(),
        )

    logger.info("Creating Synchronizer")
    syncr = Synchronizer(streams)

    #### Basic code for interfacing with in-progress RealTimeTriangulator
    #### Just run off of saved point_data.csv for development/testing
    camera_array: CameraArray = config.get_camera_array()
    sync_packet_triangulator = SyncPacketTriangulator(
        camera_array, syncr, recording_directory=session_path
    )

    for port, stream in streams.items():
        stream.play_video()

    while sync_packet_triangulator.running:
        logger.info("Waiting for streams to finish playing")
        time.sleep(1)

    # only getting here if the things runs
    assert True


if __name__ == "__main__":
    test_hand_tracker()
