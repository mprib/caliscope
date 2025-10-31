"""
NOTE: In addition to testing the real time triangulation, this code bases
its final assertions on the values from the original bundle adjustment.
This allows a cross check for the triangulation function that is distinct from
the optimization in the bundle adjustmnent.

After recent inclusion of distortion into the triangulation, the tolerance
of the final averaged triangulated position improved from 1.5 cm to 1.5 mm.
"""

import logging
import shutil
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd

from caliscope import __root__
from caliscope.calibration.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray
from caliscope.cameras.synchronizer import Synchronizer
from caliscope.configurator import Configurator
from caliscope.helper import copy_contents
from caliscope.recording.recorded_stream import RecordedStream
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.triangulate.sync_packet_triangulator import SyncPacketTriangulator

logger = logging.getLogger(__name__)


def test_triangulator():
    original_session = Path(__root__, "tests", "sessions", "post_optimization")
    test_session = Path(
        original_session.parent.parent,
        "sessions_copy_delete",
        "post_optimization",
    )

    # clear previous test so as not to pollute current test results
    if test_session.exists() and test_session.is_dir():
        shutil.rmtree(test_session)

    copy_contents(original_session, test_session)

    recording_directory = Path(test_session, "calibration", "extrinsic")
    target_xyz_path = recording_directory / "xyz_CHARUCO.csv"

    # there should be nothing here... this is what we are going to create
    assert not target_xyz_path.exists()

    config = Configurator(test_session)

    charuco: Charuco = config.get_charuco()
    charuco_tracker = CharucoTracker(charuco)

    camera_array: CameraArray = config.get_camera_array()

    logger.info("Creating RecordedStreamPool based on calibration recordings")

    streams = {}
    for port, camera in camera_array.cameras.items():
        rotation_count = camera.rotation_count
        streams[port] = RecordedStream(
            recording_directory, port, rotation_count, fps_target=100, tracker=charuco_tracker
        )

    logger.info("Creating Synchronizer")
    syncr = Synchronizer(streams)

    #### Basic code for interfacing with in-progress RealTimeTriangulator
    #### Just run off of saved point_data.csv for development/testing
    real_time_triangulator = SyncPacketTriangulator(
        camera_array,
        syncr,
        recording_directory=recording_directory,
        tracker_name=charuco_tracker.name,
    )

    for port, stream in streams.items():
        stream.play_video()

    while real_time_triangulator.running:
        sleep(1)

    # 1. LOAD DATA
    xyz_history = pd.read_csv(target_xyz_path)

    # Load the ground truth data from the bundle adjustment (PointEstimates)
    point_estimates = config.load_point_estimates_from_toml()

    # 2. CONSTRUCT GROUND TRUTH DATAFRAME
    # Create a DataFrame from the bundle adjustment results for easier comparison.
    xyz_config_coords = point_estimates.obj
    df_ground_truth = pd.DataFrame(
        {
            "sync_index": point_estimates.sync_indices,
            "point_id": point_estimates.point_id,
            "x_coord": xyz_config_coords[point_estimates.obj_indices, 0],
            "y_coord": xyz_config_coords[point_estimates.obj_indices, 1],
            "z_coord": xyz_config_coords[point_estimates.obj_indices, 2],
        }
    )

    # Ensure we are only comparing unique 3D points
    df_ground_truth.drop_duplicates(subset=["sync_index", "point_id"], inplace=True)

    # 3. NORMALIZE SYNC INDICES
    # Normalize the ground truth sync_index to be zero-based, matching the
    # real-time triangulation output, which does not preserve the original frame index.
    min_sync_index = df_ground_truth["sync_index"].min()
    logger.info(f"Normalizing ground truth sync_index by subtracting offset of {min_sync_index}")
    df_ground_truth["sync_index"] = df_ground_truth["sync_index"] - min_sync_index

    # 4. MERGE FOR ALIGNMENT
    # Perform an inner merge to align points that appear in BOTH datasets for each frame.
    df_merged = pd.merge(
        df_ground_truth, xyz_history, on=["sync_index", "point_id"], suffixes=("_truth", "_triangulated")
    )

    # 5. CALCULATE PER-POINT ERROR
    xyz_truth = df_merged[["x_coord_truth", "y_coord_truth", "z_coord_truth"]].values
    xyz_triangulated = df_merged[["x_coord_triangulated", "y_coord_triangulated", "z_coord_triangulated"]].values
    errors_m = np.linalg.norm(xyz_truth - xyz_triangulated, axis=1)

    # 6. ENHANCED ASSERTIONS
    mean_error_mm = np.mean(errors_m) * 1000
    max_error_mm = np.max(errors_m) * 1000
    logger.info(f"Mean per-point error: {mean_error_mm:.2f} mm")
    logger.info(f"Max per-point error:  {max_error_mm:.2f} mm")

    assert mean_error_mm < 1.5  # Assert mean error is less than 1.5mm
    assert max_error_mm < 15  # Assert no single point deviates by more than 1.5cm


if __name__ == "__main__":
    import caliscope.logger

    caliscope.logger.setup_logging()
    test_triangulator()
