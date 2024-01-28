import caliscope.logger

import pandas as pd
from caliscope import __root__
import shutil
from pathlib import Path
import time
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.calibration.charuco import Charuco
import shutil
import pandas as pd
import numpy as np

logger = caliscope.logger.get(__name__)


def assert_almost_equal(val1, val2, delta, msg):
    assert abs(val1 - val2) <= delta, msg

def test_sync_stream_manager():
    original_workspace = Path(__root__, "tests", "sessions", "4_cam_recording")
    test_workspace = Path(__root__, "tests", "sessions_copy_delete", "4_cam_recording")

    copy_contents(original_workspace, test_workspace)


    config = Configurator(test_workspace)
    charuco = config.get_charuco()
    tracker = CharucoTracker(charuco)
    camera_array = config.get_camera_array()
    # tracker = None
    # all_camera_data = camera_array.cameras
    recording_dir = Path(test_workspace, "calibration", "extrinsic")

    # delete frame time history to assess success of imputed frame time method
    frame_history = Path(recording_dir, "frame_time_history.csv")
    frame_history.unlink()
        
    sync_stream_manager = SynchronizedStreamManager(
        recording_dir=recording_dir,
        all_camera_data=camera_array.cameras,
        tracker=tracker,
    )

    # playback streams with high fps target to speed processing
    sync_stream_manager.process_streams(fps_target=100)


    # Load the gold standard and test CSV files
    gold_standard_df = pd.read_csv(Path(original_workspace, "calibration", "extrinsic", "xy.csv"))
    test_data_path =  Path(test_workspace, "calibration", "extrinsic", "CHARUCO", "xy_CHARUCO.csv")

    while not test_data_path.exists():
        # wait for the tracked points to be created to compare
        logger.info("Waiting for ")
        time.sleep(1)
    
    test_df = pd.read_csv(test_data_path)
    # Adjust sync_index in gold_standard_df to start at 1
    gold_standard_df['sync_index'] -= (gold_standard_df['sync_index'].min() - 1)

    # Merge the dataframes on sync_index, port, point_id
    merged_df = pd.merge(gold_standard_df, test_df, on=['sync_index', 'port', 'point_id'], suffixes=('_gold', '_test'))

    # Define pixel tolerance for img_loc_x and img_loc_y
    pixel_tolerance = 10

    # Calculate the absolute differences and average difference for each row
    merged_df['x_diff'] = (merged_df['img_loc_x_gold'] - merged_df['img_loc_x_test']).abs()
    merged_df['y_diff'] = (merged_df['img_loc_y_gold'] - merged_df['img_loc_y_test']).abs()
    mean_x_diff = merged_df["x_diff"].mean()
    assert abs(mean_x_diff) < pixel_tolerance
    mean_y_diff = merged_df["y_diff"].mean()
    assert abs(mean_y_diff) < pixel_tolerance

    logger.info(f"Mean x difference is {mean_x_diff} pixels")
    logger.info(f"Mean y difference is {mean_y_diff} pixels")


    
if __name__ == "__main__":
    test_sync_stream_manager()
