import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from caliscope import __root__
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.synchronized_stream_manager import SynchronizedStreamManager

logger = logging.getLogger(__name__)


def test_synchronizer(tmp_path: Path):
    original_session_path = Path(__root__, "tests", "sessions", "4_cam_recording")

    # clear previous test so as not to pollute current test results
    if tmp_path.exists() and tmp_path.is_dir():
        logger.info(f"Removing previously copied sessions at {tmp_path}")
        shutil.rmtree(tmp_path)

    logger.info(f"Copying over files from {original_session_path} to {tmp_path} for testing purposes")
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    config = Configurator(tmp_path)

    logger.info("Creating RecordedStreamPool")
    recording_directory = Path(tmp_path, "recordings", "recording_1")

    camera_array = config.get_camera_array()
    stream_manager = SynchronizedStreamManager(recording_dir=recording_directory, all_camera_data=camera_array.cameras)

    logger.info("Creating Synchronizer")
    stream_manager.process_streams(fps_target=100)
    target_frame_time_path = Path(recording_directory, "processed", "frame_time_history.csv")

    while not target_frame_time_path.exists():
        # recorder hasn't finished yet
        time.sleep(1)

    df = pd.read_csv(target_frame_time_path)

    # Group by sync_index and calculate the min and max frame_time for each group
    group = df.groupby("sync_index")["frame_time"].agg(["min", "max"])
    logger.info(group)

    # Iterate over consecutive pairs of groups
    for i in range(len(group) - 1):
        # Check that the max frame_time of the current group is less than the min frame_time of the next group
        assert group["max"].iloc[i] < group["min"].iloc[i + 1]


if __name__ == "__main__":
    test_synchronizer()
