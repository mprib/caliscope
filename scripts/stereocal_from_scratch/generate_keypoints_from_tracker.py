"""
Integration script demonstrating ArUcoTracker with SynchronizedStreamManager.

This is a working reference implementation that will be purged once
ArUcoTracker is fully integrated into the main workflow.
"""

import logging
import time
import shutil
from pathlib import Path

from caliscope import __root__
from caliscope.configurator import Configurator
from caliscope.logger import setup_logging
from caliscope.synchronized_stream_manager import SynchronizedStreamManager
from caliscope.tracker import Tracker
from caliscope.trackers.aruco_tracker import ArucoTracker
from caliscope.helper import copy_contents

setup_logging()
logger = logging.getLogger(__name__)


def generate_keypoints(project_dir: Path, tracker: Tracker):
    """Process calibration videos with ArUcoTracker to create xy_ARUCO.csv."""

    raw_video_dir = project_dir / "calibration/extrinsic"

    # Load camera array from config
    config = Configurator(project_dir)
    camera_array = config.get_camera_array()

    logger.info(f"Processing calibration videos with {tracker.name} tracker...")

    # Create synchronized stream manager
    sync_stream_manager = SynchronizedStreamManager(
        recording_dir=raw_video_dir, all_camera_data=camera_array.cameras, tracker=tracker
    )

    # Process streams (ArUco detection is fast, so high fps_target is fine)
    sync_stream_manager.process_streams(fps_target=100, include_video=True)
    # Wait for output file to be created
    target_output_file = project_dir / f"calibration/extrinsic/{tracker.name}/xy_{tracker.name}.csv"

    timeout = 30  # seconds
    start_time = time.time()
    while not target_output_file.exists():
        if time.time() - start_time > timeout:
            logger.error(f"Timeout waiting for output file: {target_output_file}")
            return
        time.sleep(0.5)
        logger.info(f"Waiting for {target_output_file}")

    assert target_output_file.exists(), "Tracker output creation failed"

    logger.info(f"Successfully created: {target_output_file}")


def generate_aruco_xy():
    # where data will come from and go
    test_data_dir = __root__ / "tests/sessions/post_optimization"
    fixture_project_dir = __root__ / "scripts/fixtures/aruco_pipeline"

    # clear out old data to ensure no cross contamination
    if fixture_project_dir.exists():
        shutil.rmtree(fixture_project_dir)

    assert not fixture_project_dir.exists()

    copy_contents(test_data_dir, fixture_project_dir)

    tracker = ArucoTracker(inverted=True)
    generate_keypoints(fixture_project_dir, tracker)


# This duplicates the default data in the test directory.
# But getting this working helped ensure the

# def generate_charuco_xy():
#     # where data will come from and go
#     test_data_dir = __root__ / "tests/sessions/post_optimization"
#     fixture_project_dir = __root__ / "scripts/fixtures/aruco_pipeline"
#
#     # clear out old data to ensure no cross contamination
#     if fixture_project_dir.exists():
#         shutil.rmtree(fixture_project_dir)
#
#     assert not fixture_project_dir.exists()
#
#     copy_contents(test_data_dir, fixture_project_dir)
#
#     config = Configurator(fixture_project_dir)
#     charuco: Charuco = config.get_charuco()
#     tracker = CharucoTracker(charuco=charuco)
#     generate_keypoints(fixture_project_dir, tracker)


if __name__ == "__main__":
    # Create ArUcoTracker with inverted=True for current test fixture
    generate_aruco_xy()
