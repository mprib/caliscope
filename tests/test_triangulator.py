""""
NOTE: In addition to testing the real time triangulation, this code bases
its final assertions on the values from the original bundle adjustment. 
This allows a cross check for the triangulation function that is distinct from 
the optimization in the bundle adjustmnent. 

After recent inclusion of distortion into the triangulation, the tolerance 
of the final averaged triangulated position improved from 1.5 cm to 6 mm.
"""
import caliscope.logger

from time import sleep

from caliscope.cameras.synchronizer import Synchronizer
from caliscope.triangulate.sync_packet_triangulator import SyncPacketTriangulator
from caliscope.cameras.camera_array import CameraArray
from caliscope.recording.recorded_stream import RecordedStream
from caliscope.calibration.charuco import Charuco
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from caliscope import __root__

logger = caliscope.logger.get(__name__)



def test_triangulator():
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
    config.refresh_point_estimates_from_toml()
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

    logger.info("Assert that mean positions are within 7 millimeters...")
    assert abs(config_x_mean - triangulator_x_mean) < 0.007
    assert abs(config_y_mean - triangulator_y_mean) < 0.007
    assert abs(config_z_mean - triangulator_z_mean) < 0.007


if __name__ == "__main__":

    test_triangulator()
# %%
