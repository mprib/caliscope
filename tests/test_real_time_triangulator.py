""""
The Place where I'm putting together the RealTimeTriangulator working stuff that should one day become a test

Hopefully I can keep things clean enough for that...

"""
# %%
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
from time import sleep

from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.interface import PointPacket, FramePacket, SyncPacket
from pyxy3d.triangulate.sync_packet_triangulator import SyncPacketTriangulator
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d.calibration.charuco import Charuco, get_charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.configurator import Configurator

import pytest
import shutil
from pathlib import Path
from numba import jit
from numba.typed import Dict, List
import numpy as np
import cv2
import pandas as pd
from time import time
from pyxy3d import __root__

TEST_SESSIONS = ["post_optimization"]


def copy_contents(src_folder, dst_folder):
    """
    Helper function to port a test case data folder over to a temp directory
    used for testing purposes so that the test case data doesn't get overwritten
    """
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)

    # Create the destination folder if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)

    for item in src_path.iterdir():
        # Construct the source and destination paths
        src_item = src_path / item
        dst_item = dst_path / item.name

        # Copy file or directory
        if src_item.is_file():
            logger.info(f"Copying file at {src_item} to {dst_item}")
            shutil.copy2(src_item, dst_item)  # Copy file preserving metadata

        elif src_item.is_dir():
            logger.info(f"Copying directory at {src_item} to {dst_item}")
            shutil.copytree(src_item, dst_item)


@pytest.fixture(params=TEST_SESSIONS)
def session_path(request, tmp_path):
    """
    Tests will be run based on data stored in tests/sessions, but to avoid overwriting
    or altering test cases,the tested directory will get copied over to a temp
    directory and then that temp directory will be passed on to the calling functions
    """
    original_test_data_path = Path(__root__, "tests", "sessions", request.param)
    tmp_test_data_path = Path(tmp_path, request.param)
    copy_contents(original_test_data_path, tmp_test_data_path)

    return tmp_test_data_path
    # return original_test_data_path


def test_real_time_triangulator(session_path):
    config = Configurator(session_path)
    # origin_sync_index = config.dict["capture_volume"]["origin_sync_index"]

    charuco: Charuco = config.get_charuco()
    charuco_tracker = CharucoTracker(charuco)

    camera_array: CameraArray = config.get_camera_array()

    logger.info(f"Creating RecordedStreamPool based on calibration recordings")
    recording_directory = Path(session_path, "calibration", "extrinsic")
    stream_pool = RecordedStreamPool(
        directory=recording_directory,
        config=config,
        tracker=charuco_tracker,
        fps_target=100,
    )
    logger.info("Creating Synchronizer")
    syncr = Synchronizer(stream_pool.streams, fps_target=100)

    #### Basic code for interfacing with in-progress RealTimeTriangulator
    #### Just run off of saved point_data.csv for development/testing
    real_time_triangulator = SyncPacketTriangulator(
        camera_array,
        syncr,
        recording_directory=recording_directory,
        tracker_name=charuco_tracker.name,
    )
    stream_pool.play_videos()
    while real_time_triangulator.running:
        sleep(1)

    # %%
    # need to compare the output of the triangulator to the point_estimats
    # this is nice because it's two totally different processing pipelines
    # but sync indices will be different, so just compare mean positions
    # which should be quite close

    xyz_history = pd.read_csv(Path(recording_directory, "xyz_CHARUCO.csv"))
    xyz_config = np.array(config.dict["point_estimates"]["obj"])
    triangulator_x_mean = xyz_history["x_coord"].mean()
    triangulator_y_mean = xyz_history["y_coord"].mean()
    triangulator_z_mean = xyz_history["z_coord"].mean()

    config_x_mean = xyz_config[:, 0].mean()
    config_y_mean = xyz_config[:, 1].mean()
    config_z_mean = xyz_config[:, 2].mean()

    logger.info(f"x: {round(triangulator_x_mean,4)} vs {round(config_x_mean,4)} ")
    logger.info(f"y: {round(triangulator_y_mean,4)} vs {round(config_y_mean,4)} ")
    logger.info(f"z: {round(triangulator_z_mean,4)} vs {round(config_z_mean,4)} ")

    logger.info(f"Assert that mean positions are within 1.5 centimeters...")
    assert abs(config_x_mean - triangulator_x_mean) < 0.015
    assert abs(config_y_mean - triangulator_y_mean) < 0.015
    assert abs(config_z_mean - triangulator_z_mean) < 0.015


if __name__ == "__main__":
    original_session_path = Path(__root__, "tests", "sessions", "post_optimization")
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        "post_monocal_post_optimization",
    )

    # clear previous test so as not to pollute current test results
    if session_path.exists() and session_path.is_dir():
        shutil.rmtree(session_path)

    copy_contents(original_session_path, session_path)

    test_real_time_triangulator(session_path)
# %%
