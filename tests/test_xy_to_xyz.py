import caliscope.logger

import time
import pandas as pd
from pathlib import Path
from caliscope import __root__
from caliscope.configurator import Configurator
# from caliscope.post_processing.post_processor import PostProcessor
from caliscope.triangulate.triangulation import triangulate_xy

from caliscope.helper import copy_contents
from caliscope.trackers.tracker_enum import TrackerEnum
logger = caliscope.logger.get(__name__)

def test_xy_to_xyz_postprocessing():
    # load in file of xy point data
    origin_data = Path(__root__, "tests", "sessions", "4_cam_recording")
    working_data = Path(__root__,"tests", "sessions_copy_delete", "4_cam_recording_2") # create alternate test directory because running into permission errors when invoking pytest
    
    copy_contents(origin_data, working_data)

    config = Configurator(working_data)
    recording_directory = Path(working_data, "recording_1")
    tracker_enum = TrackerEnum.HOLISTIC

    xy_path = Path(recording_directory,tracker_enum.name, f"xy_{tracker_enum.name}.csv")
    xy_data = pd.read_csv(xy_path)

    start = time.time()
    logger.info(f"beginning triangulation at {time.time()}")

    # note: triangulate_xy  is a method used primarily internally by the PostProcessor
    # the method create_xyz uses it.
    camera_array = config.get_camera_array()

    xyz_history = triangulate_xy(xy_data,camera_array) 
    logger.info(f"ending triangulation at {time.time()}")
    stop = time.time()
    logger.info(f"Elapsed time is {stop-start}. Note that on first iteration, @jit functions will take longer")

    # Assert that the xyz_history dictionary has the expected keys
    assert set(xyz_history.keys()) == {"sync_index", "point_id", "x_coord", "y_coord", "z_coord"}

    # Assert that all lists in xyz_history have the same length
    assert (
        len(xyz_history["sync_index"])
        == len(xyz_history["point_id"])
        == len(xyz_history["x_coord"])
        == len(xyz_history["y_coord"])
        == len(xyz_history["z_coord"])
    )

    # Assert that coordinates are within expected bounds around the origin
    min_x, max_x = -2, 2    
    min_y, max_y = -2, 2    
    min_z, max_z = -2, 4    
    for x, y, z in zip(xyz_history["x_coord"], xyz_history["y_coord"], xyz_history["z_coord"]):
        assert min_x <= x <= max_x
        assert min_y <= y <= max_y
        assert min_z <= z <= max_z


    output_path = Path(recording_directory, "xyz.csv")
    xyz_history = pd.DataFrame(xyz_history)
    xyz_history.to_csv(output_path)

if __name__ == "__main__":
    test_xy_to_xyz_postprocessing()
