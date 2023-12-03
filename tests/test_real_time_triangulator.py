""""
The Place where I'm putting together the RealTimeTriangulator working stuff that should one day become a test

Hopefully I can keep things clean enough for that...

"""
import pyxy3d.logger

from time import sleep

from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.triangulate.sync_packet_triangulator import SyncPacketTriangulator
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.configurator import Configurator
from pyxy3d.helper import copy_contents

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from pyxy3d import __root__

logger = pyxy3d.logger.get(__name__)



def test_real_time_triangulator():
    original_session = Path(__root__, "tests", "sessions", "post_optimization")
    test_session = Path(
        original_session.parent.parent,
        "sessions_copy_delete",
        "post_optimization",
    )

    # clear previous test so as not to pollute current test results
    if test_session.exists() and test_session.is_dir():
        shutil.rmtree(test_session)

    copy_contents(original_session, test_session)
    
    config = Configurator(test_session)
    # origin_sync_index = config.dict["capture_volume"]["origin_sync_index"]

    charuco: Charuco = config.get_charuco()
    charuco_tracker = CharucoTracker(charuco)

    camera_array: CameraArray = config.get_camera_array()

    logger.info("Creating RecordedStreamPool based on calibration recordings")
    recording_directory = Path(test_session, "calibration", "extrinsic")
    
    streams = {}
    for port, camera in camera_array.cameras.items():
        rotation_count = camera.rotation_count
        streams[port] = RecordedStream(
            recording_directory,
            port,
            rotation_count,
            fps_target=100,
            tracker=charuco_tracker
        )
    
    logger.info("Creating Synchronizer")
    syncr = Synchronizer(streams)

    #### Basic code for interfacing with in-progress RealTimeTriangulator
    #### Just run off of saved point_data.csv for development/testing
    real_time_triangulator = SyncPacketTriangulator(
        camera_array,
        syncr,
        recording_directory=recording_directory,
        tracker_name=charuco_tracker.name,
    )
    
    for port, stream in streams.items():
        stream.play_video()

    while real_time_triangulator.running:
        sleep(1)

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

    logger.info("Assert that mean positions are within 1.5 centimeters...")
    assert abs(config_x_mean - triangulator_x_mean) < 0.015
    assert abs(config_y_mean - triangulator_y_mean) < 0.015
    assert abs(config_z_mean - triangulator_z_mean) < 0.015


if __name__ == "__main__":

    test_real_time_triangulator()
# %%
