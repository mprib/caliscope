"""
Integration script demonstrating ArUcoTracker with SynchronizedStreamManager.

This is a working reference implementation that will be purged once
ArUcoTracker is fully integrated into the main workflow.
"""

import logging
import time
import shutil

from caliscope import __root__
from caliscope.configurator import Configurator
from caliscope.logger import setup_logging
from caliscope.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.trackers.aruco_tracker import ArucoTracker
from caliscope.helper import copy_contents

setup_logging()
logger = logging.getLogger(__name__)


def generate_aruco_keypoints():
    """Process calibration videos with ArUcoTracker to create xy_ARUCO.csv."""

    # Use post_optimization test data (has calibration videos)
    test_data_dir = __root__ / "tests/sessions/post_optimization"
    calibration_video_dir = test_data_dir / "calibration/extrinsic"
    fixture_dir = __root__ / "scripts/fixtures/aruco_pipeline"

    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)

    copy_contents(test_data_dir, fixture_dir)

    if not calibration_video_dir.exists():
        logger.error(f"Calibration video directory not found: {calibration_video_dir}")
        return

    # Load camera array from config
    config = Configurator(fixture_dir)
    camera_array = config.get_camera_array()

    # Create ArUcoTracker with inverted=True for current test fixture
    # TODO: Update test fixture to use normal markers and set inverted=False
    aruco_tracker = ArucoTracker(inverted=True)

    logger.info(f"Processing calibration videos with {aruco_tracker.name} tracker...")

    raw_video_dir = fixture_dir / "calibration/extrinsic"
    # Create synchronized stream manager
    sync_stream_manager = SynchronizedStreamManager(
        recording_dir=raw_video_dir, all_camera_data=camera_array.cameras, tracker=aruco_tracker
    )

    # Process streams (ArUco detection is fast, so high fps_target is fine)
    sync_stream_manager.process_streams(fps_target=100, include_video=True)
    # Wait for output file to be created
    target_output_file = fixture_dir / f"calibration/extrinsic/{aruco_tracker.name}/xy_{aruco_tracker.name}.csv"

    timeout = 30  # seconds
    start_time = time.time()
    while not target_output_file.exists():
        if time.time() - start_time > timeout:
            logger.error(f"Timeout waiting for output file: {target_output_file}")
            return
        time.sleep(0.5)
        logger.info(f"Waiting for {target_output_file}")

    logger.info(f"Successfully created: {target_output_file}")

    # Validate output format
    import pandas as pd

    df = pd.read_csv(target_output_file)

    required_columns = [
        "sync_index",
        "port",
        "frame_index",
        "frame_time",
        "point_id",
        "img_loc_x",
        "img_loc_y",
        "obj_loc_x",
        "obj_loc_y",
    ]

    assert all(col in df.columns for col in required_columns), "Missing required columns"
    assert df["obj_loc_x"].isna().all() and df["obj_loc_y"].isna().all(), "obj_loc should be empty"

    # Verify point ID pattern
    sample_ids = df["point_id"].unique()
    assert all(id % 10 in range(1, 5) for id in sample_ids), "Point IDs don't follow corner pattern"

    logger.info(
        f"Output validation passed. Detected {len(sample_ids)} unique point IDs  \
        across {df['sync_index'].nunique()} frames."
    )
    logger.info(f"Sample point IDs: {sorted(sample_ids)[:10]}")


if __name__ == "__main__":
    generate_aruco_keypoints()
