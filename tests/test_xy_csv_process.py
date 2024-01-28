
import caliscope.logger

import sys
from caliscope.configurator import Configurator
from pathlib import Path
from caliscope import __root__
import pandas as pd

# specify a source directory (with recordings)
from caliscope.helper import copy_contents
from caliscope.post_processing.post_processor import PostProcessor
from caliscope.trackers.tracker_enum import TrackerEnum

logger = caliscope.logger.get(__name__)


def test_xy_point_creation():
    # create a clean directory to start from
    session_path = Path(__root__, "tests", "sessions", "mediapipe_calibration_2_cam")
    copy_session_path = Path(
        __root__, "tests", "sessions_copy_delete", "mediapipe_calibration_2_cam"
    )
    copy_contents(session_path, copy_session_path)

    # create inputs to processing pipeline function
    # config = Configurator(copy_session_path)

    config = Configurator(copy_session_path)
    camera_array = config.get_camera_array()
    recording_path = Path(copy_session_path, "recordings", "recording_1")
    tracker_enum = TrackerEnum.HAND
    post_processor = PostProcessor(
        camera_array=camera_array,
        recording_path=recording_path,
        tracker_enum=tracker_enum,
    )

    # make some basic assertions against the created files
    produced_files = [
        Path(recording_path, "HAND", "xy_HAND.csv"),
        Path(recording_path, "HAND", "port_0_HAND.mp4"),
        Path(recording_path, "HAND", "port_1_HAND.mp4"),
    ]

    # confirm that the directory does not have these files prior to running xy creation method
    for file in produced_files:
        logger.info(f"Asserting that the following file exists: {file}")
        assert not file.exists()

    post_processor.create_xy()
    # create_xy(config, recording_directory,tracker_enum=tracker_enum)

    for file in produced_files:
        logger.info(f"Asserting that the following file exists: {file}")
        assert file.exists()

    # confirm that xy data is produced for the sync indices (slightly reduced to avoid missing data issues)
    xy_data = pd.read_csv(Path(recording_path, "HAND", f"xy_{tracker_enum.name}.csv"))
    xy_sync_index_count = xy_data["sync_index"].max() + 1  # zero indexed

    frame_times = pd.read_csv(Path(recording_path, "frame_time_history.csv"))
    sync_index_count = len(frame_times["sync_index"].unique())
    logger.info(
        f"Sync index count in frame history: {sync_index_count} in frame history"
    )
    logger.info(f"Max sync index: {xy_data['sync_index'].max()} in xy.csv")

    LEEWAY = 2  # sync indices that might not get copied over due to not enough frames
    assert sync_index_count - LEEWAY <= xy_sync_index_count


if __name__ == "__main__":
    test_xy_point_creation()
