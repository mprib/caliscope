import time
from pathlib import Path

import pandas as pd

import caliscope.logger
from caliscope import __root__
from caliscope.helper import copy_contents
from caliscope.trackers.tracker_enum import TrackerEnum

# from caliscope.post_processing.post_processor import PostProcessor
from caliscope.triangulate.triangulation import triangulate_from_files

logger = caliscope.logger.get(__name__)


def test_xy_to_xyz_postprocessing():
    # load in file of xy point data
    origin_data = Path(__root__, "tests", "sessions", "4_cam_recording")
    working_data = Path(
        __root__, "tests", "sessions_copy_delete", "4_cam_recording_2"
    )  # create alternate test directory because running into permission errors when invoking pytest

    copy_contents(origin_data, working_data)


    config_path = Path(working_data,"config.toml")
    recording_directory = Path(working_data, "recordings", "recording_1")
    tracker_enum = TrackerEnum.HOLISTIC

    xy_path = Path(recording_directory, tracker_enum.name, f"xy_{tracker_enum.name}.csv")

    start = time.time()
    logger.info(f"beginning triangulation at {time.time()}")

    xyz_recalculated = triangulate_from_files(config_path,xy_path)

    logger.info(f"ending triangulation at {time.time()}")
    stop = time.time()
    logger.info(f"Elapsed time is {stop-start}. Note that on first iteration, @jit functions will take longer")

    # load in previously triangulated data
    original_xyz_path = Path(xy_path.parent, f"xyz_{tracker_enum.name}.csv")
    original_xyz = pd.read_csv(original_xyz_path)

    # After loading original_xyz and calculating xyz_recalculated:

    # Filter both datasets to only include face points, which have point_ids >= 500
    # other points moved in and out of view causing more jitter that was smoothed
    # with downstream filtering in the original triangulation process.
    original_xyz_face = original_xyz[original_xyz['point_id'] >= 500]
    xyz_recalculated_face = xyz_recalculated[xyz_recalculated['point_id'] >= 500]

    # Reset indices after filtering
    original_xyz_face = original_xyz_face.reset_index(drop=True)
    xyz_recalculated_face = xyz_recalculated_face.reset_index(drop=True)

    # Remove index column if it exists
    original_xyz_face = original_xyz_face.drop('Unnamed: 0', axis=1, errors='ignore')

    # Make sure both filtered dataframes have the same shape
    assert original_xyz_face.shape == xyz_recalculated_face.shape, (
        f"Shape mismatch: original {original_xyz_face.shape}, recalculated {xyz_recalculated_face.shape}")

    # Sort both dataframes by sync_index and point_id to ensure they're aligned
    original_xyz_face = original_xyz_face.sort_values(['sync_index', 'point_id']).reset_index(drop=True)
    xyz_recalculated_face = xyz_recalculated_face.sort_values(['sync_index', 'point_id']).reset_index(drop=True)

    # Define acceptable tolerance for floating point comparisons
    tolerance = 0.015  # Note that the original data has been filtered and smoothed...this is just raw triangulated data

    # Compare coordinates with tolerance
    coord_diff_x = abs(original_xyz_face['x_coord'] - xyz_recalculated_face['x_coord'])
    coord_diff_y = abs(original_xyz_face['y_coord'] - xyz_recalculated_face['y_coord'])
    coord_diff_z = abs(original_xyz_face['z_coord'] - xyz_recalculated_face['z_coord'])

    # Assert maximum differences are within tolerance
    assert coord_diff_x.max() < tolerance, (
        f"Maximum x coordinate difference {coord_diff_x.max()} exceeds tolerance {tolerance}"
        )

    assert coord_diff_y.max() < tolerance, (
        f"Maximum y coordinate difference {coord_diff_y.max()} exceeds tolerance {tolerance}"
        )

    assert coord_diff_z.max() < tolerance, (
        f"Maximum z coordinate difference {coord_diff_z.max()} exceeds tolerance {tolerance}"
        )

    # Additional test: Verify that at least 95% of points are very close (stricter tolerance)
    strict_tolerance = 0.005  # Stricter tolerance for most points
    pct_close_x = (coord_diff_x < strict_tolerance).mean() * 100
    pct_close_y = (coord_diff_y < strict_tolerance).mean() * 100
    pct_close_z = (coord_diff_z < strict_tolerance).mean() * 100

    assert pct_close_x >= 95, f"Only {pct_close_x:.1f}% of x coordinates are within strict tolerance"
    assert pct_close_y >= 95, f"Only {pct_close_y:.1f}% of y coordinates are within strict tolerance"
    assert pct_close_z >= 95, f"Only {pct_close_z:.1f}% of z coordinates are within strict tolerance"


    # Log the maximum differences to help debug
    logger.info(
        f"Maximum coordinate differences - X: {coord_diff_x.max()}, Y: {coord_diff_y.max()}, Z: {coord_diff_z.max()}"
        )
    output_path = Path(recording_directory, "xyz.csv")
    xyz_recalculated = pd.DataFrame(xyz_recalculated)
    xyz_recalculated.to_csv(output_path)



if __name__ == "__main__":
    test_xy_to_xyz_postprocessing()
