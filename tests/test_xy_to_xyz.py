import logging
import time
from pathlib import Path

import pandas as pd
from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.point_data import ImagePoints, WorldPoints
from caliscope import __root__
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.trackers.tracker_enum import TrackerEnum
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope import persistence

# from caliscope.post_processing.post_processor import PostProcessor
from caliscope.triangulate.triangulation import triangulate_from_files


logger = logging.getLogger(__name__)


def test_image_points_to_world_points(tmp_path: Path):
    # load in file of xy point data
    origin_data = Path(__root__, "tests", "sessions", "4_cam_recording")

    copy_contents_to_clean_dest(origin_data, tmp_path)

    recording_directory = Path(tmp_path, "recordings", "recording_1")
    tracker_enum = TrackerEnum.HOLISTIC

    xy_path = Path(recording_directory, tracker_enum.name, f"xy_{tracker_enum.name}.csv")

    # load in previously triangulated data
    original_xyz_path = Path(xy_path.parent, f"xyz_{tracker_enum.name}.csv")
    original_xyz = pd.read_csv(original_xyz_path)

    image_points = ImagePoints.from_csv(xy_path)
    camera_array: CameraArray = persistence.load_camera_array(tmp_path / "camera_array.toml")

    start = time.time()
    logger.info(f"beginning triangulation at {time.time()}")

    world_points: WorldPoints = image_points.triangulate(camera_array=camera_array)

    logger.info(f"ending triangulation at {time.time()}")
    stop = time.time()
    logger.info(
        f"Elapsed time is {stop - start:.1f} seconds. Note that on first iteration, @jit functions will take longer"
    )

    # Verify provenance is tracked
    assert world_points._source_image_points is image_points, "Source ImagePoints not tracked"
    assert world_points._camera_array is camera_array, "CameraArray not tracked"

    # Verify conversion method exists and works
    point_estimates = world_points.to_point_estimates(image_points, camera_array)
    assert isinstance(point_estimates, PointEstimates), "to_point_estimates() should return PointEstimates"

    xyz_recalculated = world_points.df

    # After loading original_xyz and calculating xyz_recalculated:
    # Filter both datasets to only include face points, which have point_ids >= 500
    # other points moved in and out of view causing more jitter that was smoothed
    # with downstream filtering in the original triangulation process.
    original_xyz_face = original_xyz[original_xyz["point_id"] >= 500]
    xyz_recalculated_face = xyz_recalculated[xyz_recalculated["point_id"] >= 500]

    # Reset indices after filtering
    original_xyz_face = original_xyz_face.reset_index(drop=True)
    xyz_recalculated_face = xyz_recalculated_face.reset_index(drop=True)

    # Remove index column if it exists
    original_xyz_face = original_xyz_face.drop("Unnamed: 0", axis=1, errors="ignore")

    # Make sure both filtered dataframes have the same shape
    assert original_xyz_face.shape == xyz_recalculated_face.shape, (
        f"Shape mismatch: original {original_xyz_face.shape}, recalculated {xyz_recalculated_face.shape}"
    )

    # Sort both dataframes by sync_index and point_id to ensure they're aligned
    original_xyz_face = original_xyz_face.sort_values(["sync_index", "point_id"]).reset_index(drop=True)
    xyz_recalculated_face = xyz_recalculated_face.sort_values(["sync_index", "point_id"]).reset_index(drop=True)

    # Define acceptable tolerance for floating point comparisons
    MAX_DEVIATION_METERS = 0.015

    # Compare coordinates with tolerance
    coord_diff_x = abs(original_xyz_face["x_coord"] - xyz_recalculated_face["x_coord"])
    coord_diff_y = abs(original_xyz_face["y_coord"] - xyz_recalculated_face["y_coord"])
    coord_diff_z = abs(original_xyz_face["z_coord"] - xyz_recalculated_face["z_coord"])

    # Assert maximum differences are within tolerance
    assert coord_diff_x.max() < MAX_DEVIATION_METERS, (
        f"Maximum x coordinate difference {coord_diff_x.max()} exceeds tolerance {MAX_DEVIATION_METERS}"
    )

    assert coord_diff_y.max() < MAX_DEVIATION_METERS, (
        f"Maximum y coordinate difference {coord_diff_y.max()} exceeds tolerance {MAX_DEVIATION_METERS}"
    )

    assert coord_diff_z.max() < MAX_DEVIATION_METERS, (
        f"Maximum z coordinate difference {coord_diff_z.max()} exceeds tolerance {MAX_DEVIATION_METERS}"
    )

    # Additional test: Verify that at least 95% of points are very close (stricter tolerance)
    STRICT_DEVIATION_METERS = 0.0025  # Stricter tolerance for most points
    pct_close_x = (coord_diff_x < STRICT_DEVIATION_METERS).mean() * 100
    pct_close_y = (coord_diff_y < STRICT_DEVIATION_METERS).mean() * 100
    pct_close_z = (coord_diff_z < STRICT_DEVIATION_METERS).mean() * 100

    assert pct_close_x >= 95, f"Only {pct_close_x:.1f}% of x coordinates are within strict tolerance"
    assert pct_close_y >= 95, f"Only {pct_close_y:.1f}% of y coordinates are within strict tolerance"
    assert pct_close_z >= 95, f"Only {pct_close_z:.1f}% of z coordinates are within strict tolerance"

    # Log the maximum differences to help debug
    logger.info(
        f"Maximum coordinate differences - X: {coord_diff_x.max()}, Y: {coord_diff_y.max()}, Z: {coord_diff_z.max()}"
    )

    logger.info(f"{pct_close_x:.1f}% of x coordinates are within strict tolerance")
    logger.info(f"{pct_close_y:.1f}% of y coordinates are within strict tolerance")
    logger.info(f"{pct_close_x:.1f}% of z coordinates are within strict tolerance")

    output_path = Path(recording_directory, "xyz.csv")
    xyz_recalculated = pd.DataFrame(xyz_recalculated)
    xyz_recalculated.to_csv(output_path)


def test_triangulate_from_files(tmp_path: Path):
    # load in file of xy point data
    origin_data = Path(__root__, "tests", "sessions", "4_cam_recording")

    copy_contents_to_clean_dest(origin_data, tmp_path)

    recording_directory = Path(tmp_path, "recordings", "recording_1")
    tracker_enum = TrackerEnum.HOLISTIC

    xy_path = Path(recording_directory, tracker_enum.name, f"xy_{tracker_enum.name}.csv")

    start = time.time()
    logger.info(f"beginning triangulation at {time.time()}")

    xyz_recalculated = triangulate_from_files(
        camera_array_path=(tmp_path / "camera_array.toml"), image_point_path=xy_path
    )

    logger.info(f"ending triangulation at {time.time()}")
    stop = time.time()

    logger.info(
        f"Elapsed time is {stop - start:.1f} seconds. Note that on first iteration, @jit functions will take longer"
    )

    # load in previously triangulated data
    original_xyz_path = Path(xy_path.parent, f"xyz_{tracker_enum.name}.csv")
    original_xyz = pd.read_csv(original_xyz_path)

    # After loading original_xyz and calculating xyz_recalculated:

    # Filter both datasets to only include face points, which have point_ids >= 500
    # other points moved in and out of view causing more jitter that was smoothed
    # with downstream filtering in the original triangulation process.
    original_xyz_face = original_xyz[original_xyz["point_id"] >= 500]
    xyz_recalculated_face = xyz_recalculated[xyz_recalculated["point_id"] >= 500]

    # Reset indices after filtering
    original_xyz_face = original_xyz_face.reset_index(drop=True)
    xyz_recalculated_face = xyz_recalculated_face.reset_index(drop=True)

    # Remove index column if it exists
    original_xyz_face = original_xyz_face.drop("Unnamed: 0", axis=1, errors="ignore")

    # Make sure both filtered dataframes have the same shape
    assert original_xyz_face.shape == xyz_recalculated_face.shape, (
        f"Shape mismatch: original {original_xyz_face.shape}, recalculated {xyz_recalculated_face.shape}"
    )

    # Sort both dataframes by sync_index and point_id to ensure they're aligned
    original_xyz_face = original_xyz_face.sort_values(["sync_index", "point_id"]).reset_index(drop=True)
    xyz_recalculated_face = xyz_recalculated_face.sort_values(["sync_index", "point_id"]).reset_index(drop=True)

    # Define acceptable tolerance for floating point comparisons
    MAX_DEVIATION_METERS = 0.015

    # Compare coordinates with tolerance
    coord_diff_x = abs(original_xyz_face["x_coord"] - xyz_recalculated_face["x_coord"])
    coord_diff_y = abs(original_xyz_face["y_coord"] - xyz_recalculated_face["y_coord"])
    coord_diff_z = abs(original_xyz_face["z_coord"] - xyz_recalculated_face["z_coord"])

    # Assert maximum differences are within tolerance
    assert coord_diff_x.max() < MAX_DEVIATION_METERS, (
        f"Maximum x coordinate difference {coord_diff_x.max()} exceeds tolerance {MAX_DEVIATION_METERS}"
    )

    assert coord_diff_y.max() < MAX_DEVIATION_METERS, (
        f"Maximum y coordinate difference {coord_diff_y.max()} exceeds tolerance {MAX_DEVIATION_METERS}"
    )

    assert coord_diff_z.max() < MAX_DEVIATION_METERS, (
        f"Maximum z coordinate difference {coord_diff_z.max()} exceeds tolerance {MAX_DEVIATION_METERS}"
    )

    # Additional test: Verify that at least 95% of points are very close (stricter tolerance)
    STRICT_DEVIATION_METERS = 0.0025  # Stricter tolerance for most points
    pct_close_x = (coord_diff_x < STRICT_DEVIATION_METERS).mean() * 100
    pct_close_y = (coord_diff_y < STRICT_DEVIATION_METERS).mean() * 100
    pct_close_z = (coord_diff_z < STRICT_DEVIATION_METERS).mean() * 100

    assert pct_close_x >= 95, f"Only {pct_close_x:.1f}% of x coordinates are within strict tolerance"
    assert pct_close_y >= 95, f"Only {pct_close_y:.1f}% of y coordinates are within strict tolerance"
    assert pct_close_z >= 95, f"Only {pct_close_z:.1f}% of z coordinates are within strict tolerance"

    # Log the maximum differences to help debug
    logger.info(
        f"Maximum coordinate differences - X: {coord_diff_x.max()}, Y: {coord_diff_y.max()}, Z: {coord_diff_z.max()}"
    )

    logger.info(f"{pct_close_x:.1f}% of x coordinates are within strict tolerance")
    logger.info(f"{pct_close_y:.1f}% of y coordinates are within strict tolerance")
    logger.info(f"{pct_close_x:.1f}% of z coordinates are within strict tolerance")

    output_path = Path(recording_directory, "xyz.csv")
    xyz_recalculated = pd.DataFrame(xyz_recalculated)
    xyz_recalculated.to_csv(output_path)


if __name__ == "__main__":
    import caliscope.logger

    caliscope.logger.setup_logging()
    temp = Path(__file__).parent / "debug"
    test_image_points_to_world_points(temp)
    test_image_points_to_world_points(temp)
