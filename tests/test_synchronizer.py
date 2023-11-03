
import pyxy3d.logger

import pandas as pd
from pyxy3d import __root__
import shutil
from pathlib import Path
import time
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.recording.recorded_stream import RecordedStreamPool
from pyxy3d.configurator import Configurator
from pyxy3d.helper import copy_contents
from pyxy3d.recording.video_recorder import VideoRecorder
logger = pyxy3d.logger.get(__name__)

# TEST_SESSIONS = ["mediapipe_calibration"]


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

    stream_pool = RecordedStreamPool(
        recording_directory,
        config=config,
        fps_target=100,
    )
    logger.info("Creating Synchronizer")
    syncr = Synchronizer(stream_pool.streams, fps_target=100)

    recorder = VideoRecorder(syncr, suffix="test")
    
    test_recordings = Path(session_path, "test_recording_1")
    recorder.start_recording(destination_folder=test_recordings, include_video=True,show_points=False, store_point_history=False)
    stream_pool.play_videos()
    target_frame_time_path = Path(test_recordings, "frame_time_history.csv")
    
    while not target_frame_time_path.exists():
        # recorder hasn't finished yet
        time.sleep(1)
        

    df = pd.read_csv(target_frame_time_path)

    # Group by sync_index and calculate the min and max frame_time for each group
    group = df.groupby('sync_index')['frame_time'].agg(['min', 'max'])
    logger.info(group)

    # Iterate over consecutive pairs of groups
    for i in range(len(group) - 1):
        # Check that the max frame_time of the current group is less than the min frame_time of the next group
        assert group['max'].iloc[i] < group['min'].iloc[i+1]

if __name__ == "__main__":
    test_synchronizer()
