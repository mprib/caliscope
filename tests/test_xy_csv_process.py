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

import sys
from pyxy3d.configurator import Configurator
from pathlib import Path
from pyxy3d import __root__
import pandas as pd
from pyxy3d.trackers.hand_tracker import HandTrackerFactory

# specify a source directory (with recordings)
from pyxy3d.helper import copy_contents
from pyxy3d.post_processing_pipelines import create_xy_points


def test_xy_point_creation():
    # create a clean directory to start from 
    session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration_2_cam")
    copy_session_path = Path(
        __root__, "tests", "sessions_copy_delete", "mediapipe_calibration_2_cam"
    )
    copy_contents(session_path, copy_session_path)
    
    # create inputs to processing pipeline function
    config = Configurator(copy_session_path)
    tracker_factory = HandTrackerFactory()
    recording_directory = Path(copy_session_path, "calibration", "extrinsic")

    frame_times = pd.read_csv(Path(recording_directory, "frame_time_history.csv"))
    sync_index_count = len(frame_times["sync_index"].unique())

    create_xy_points(config, recording_directory, tracker_factory)


    # make some basic assertions against the created files
    produced_files = [
        Path(recording_directory, "xy.csv"),
        Path(recording_directory, "port_0_xy.mp4"),
        Path(recording_directory, "port_1_xy.mp4"),
    ]
    for file in produced_files:
        logger.info(f"Asserting that the following file exists: {file}")
        assert file.exists()

    # confirm that xy data is produced for the sync indices (slightly reduced to avoid missing data issues)
    xy_data = pd.read_csv(Path(recording_directory, "xy.csv"))
    xy_sync_index_count = xy_data["sync_index"].max() + 1 # zero indexed

    logger.info(f"Sync index count in frame history: {sync_index_count} in frame history")
    logger.info(f"Max sync index: {xy_data['sync_index'].max()} in xy.csv")

    LEEWAY = 2 # sync indices that might not get copied over due to not enough frames
    assert(sync_index_count-LEEWAY <= xy_sync_index_count)


if __name__ == "__main__":
    
    test_xy_point_creation()