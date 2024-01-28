import caliscope.logger

import pandas as pd
from caliscope import __root__
import shutil
from pathlib import Path
import time
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents
from caliscope.synchronized_stream_manager import SynchronizedStreamManager

logger = caliscope.logger.get(__name__)

def test_synchronizer():
    original_session_path = Path(__root__, "tests", "sessions", "4_cam_recording")
    session_path = Path(
        original_session_path.parent.parent,
        "sessions_copy_delete",
        "synchronizer_test",
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
    recording_directory = Path(session_path, "recording_1")

    camera_array = config.get_camera_array()
    stream_manager = SynchronizedStreamManager(
        recording_dir=recording_directory, all_camera_data=camera_array.cameras
    )

    logger.info("Creating Synchronizer")
    stream_manager.process_streams(fps_target=100)
    target_frame_time_path = Path(
        recording_directory, "processed", "frame_time_history.csv"
    )

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
