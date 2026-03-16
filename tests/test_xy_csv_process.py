import logging
from pathlib import Path

import pandas as pd

from caliscope import __root__

# specify a source directory (with recordings)
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.reconstruction.reconstructor import Reconstructor
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.charuco import Charuco
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.trackers import tracker_registry

logger = logging.getLogger(__name__)


def test_xy_point_creation(tmp_path: Path):
    # create a clean directory to start from
    session_path = Path(__root__, "tests", "sessions", "charuco_calibration_2_cam")
    copy_contents_to_clean_dest(session_path, tmp_path)

    camera_array = CameraArray.from_toml(tmp_path / "camera_array.toml")

    # Register CharucoTracker so Reconstructor can create it by name
    charuco = Charuco.from_toml(tmp_path / "charuco.toml")
    tracker_registry.register("CHARUCO", lambda: CharucoTracker(charuco), display_name="Charuco")

    recording_path = Path(tmp_path, "recordings", "recording_1")
    tracker_name = "CHARUCO"
    reconstructor = Reconstructor(
        camera_array=camera_array,
        recording_path=recording_path,
        tracker_name=tracker_name,
    )

    # make some basic assertions against the created files
    produced_files = [
        Path(recording_path, "CHARUCO", "xy_CHARUCO.csv"),
        Path(recording_path, "CHARUCO", "cam_0_CHARUCO.mp4"),
        Path(recording_path, "CHARUCO", "cam_1_CHARUCO.mp4"),
    ]

    # confirm that the directory does not have these files prior to running xy creation method
    for file in produced_files:
        logger.info(f"Asserting that the following file exists: {file}")
        assert not file.exists()

    reconstructor.create_xy()

    for file in produced_files:
        logger.info(f"Asserting that the following file exists: {file}")
        assert file.exists()

    # confirm that xy data is produced for the sync indices (slightly reduced to avoid missing data issues)
    xy_data = pd.read_csv(Path(recording_path, "CHARUCO", f"xy_{tracker_name}.csv"))
    xy_sync_index_count = xy_data["sync_index"].max() + 1  # zero indexed

    frame_timestamps = pd.read_csv(Path(recording_path, "timestamps.csv"))
    sync_index_count = len(frame_timestamps["sync_index"].unique())
    logger.info(f"Sync index count in frame timestamps: {sync_index_count}")
    logger.info(f"Max sync index: {xy_data['sync_index'].max()} in xy.csv")

    LEEWAY = 2  # sync indices that might not get copied over due to not enough frames
    assert sync_index_count - LEEWAY <= xy_sync_index_count


if __name__ == "__main__":
    import caliscope.logger

    caliscope.logger.setup_logging()

    temp_path = Path(__file__).parent / "debug"
    test_xy_point_creation(temp_path)
