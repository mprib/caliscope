import logging
import time
from pathlib import Path

import pandas as pd
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope import __root__
from caliscope.helper import copy_contents_to_clean_dest

from caliscope.triangulate.triangulation import triangulate_from_files


logger = logging.getLogger(__name__)

# Post-optimization session has both calibrated camera_array.toml and charuco xy/xyz CSVs
POST_OPT_SESSION = Path(__root__, "tests", "sessions", "post_optimization")
CHARUCO_DATA_DIR = POST_OPT_SESSION / "calibration" / "extrinsic" / "CHARUCO"


def test_image_points_to_world_points(tmp_path: Path):
    # Use post_optimization charuco data for triangulation regression testing
    copy_contents_to_clean_dest(POST_OPT_SESSION, tmp_path)

    xy_path = tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"

    # load in previously triangulated data as the reference
    original_xyz_path = tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xyz_CHARUCO.csv"
    original_xyz = pd.read_csv(original_xyz_path)

    image_points = ImagePoints.from_csv(xy_path)
    camera_array: CameraArray = CameraArray.from_toml(tmp_path / "camera_array.toml")

    start = time.time()
    logger.info(f"beginning triangulation at {time.time()}")

    world_points: WorldPoints = image_points.triangulate(camera_array=camera_array)

    logger.info(f"ending triangulation at {time.time()}")
    stop = time.time()
    logger.info(f"Elapsed time is {stop - start:.1f} seconds.")

    xyz_recalculated = world_points.df

    # Select common columns for comparison (charuco data has fewer points, no face-point filter needed)
    original_xyz = original_xyz.drop("Unnamed: 0", axis=1, errors="ignore")
    common_cols = ["sync_index", "point_id", "x_coord", "y_coord", "z_coord"]
    original_xyz_filtered = original_xyz[common_cols]
    xyz_recalculated_filtered = xyz_recalculated[common_cols]

    # Sort both dataframes by sync_index and point_id to ensure they're aligned
    original_xyz_filtered = original_xyz_filtered.sort_values(["sync_index", "point_id"]).reset_index(drop=True)
    xyz_recalculated_filtered = xyz_recalculated_filtered.sort_values(["sync_index", "point_id"]).reset_index(drop=True)

    # Make sure both filtered dataframes have the same shape
    assert original_xyz_filtered.shape == xyz_recalculated_filtered.shape, (
        f"Shape mismatch: original {original_xyz_filtered.shape}, recalculated {xyz_recalculated_filtered.shape}"
    )

    # Define acceptable tolerance for floating point comparisons
    MAX_DEVIATION_METERS = 0.015

    # Compare coordinates with tolerance
    coord_diff_x = abs(original_xyz_filtered["x_coord"] - xyz_recalculated_filtered["x_coord"])
    coord_diff_y = abs(original_xyz_filtered["y_coord"] - xyz_recalculated_filtered["y_coord"])
    coord_diff_z = abs(original_xyz_filtered["z_coord"] - xyz_recalculated_filtered["z_coord"])

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
    logger.info(f"{pct_close_z:.1f}% of z coordinates are within strict tolerance")

    output_path = tmp_path / "xyz_recalculated.csv"
    xyz_recalculated = pd.DataFrame(xyz_recalculated)
    xyz_recalculated.to_csv(output_path)


def test_triangulate_from_files(tmp_path: Path):
    # Use post_optimization charuco data for triangulation regression testing
    copy_contents_to_clean_dest(POST_OPT_SESSION, tmp_path)

    xy_path = tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"

    start = time.time()
    logger.info(f"beginning triangulation at {time.time()}")

    xyz_recalculated = triangulate_from_files(
        camera_array_path=(tmp_path / "camera_array.toml"), image_point_path=xy_path
    )

    logger.info(f"ending triangulation at {time.time()}")
    stop = time.time()

    logger.info(f"Elapsed time is {stop - start:.1f} seconds.")

    # load in previously triangulated data as the reference
    original_xyz_path = tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xyz_CHARUCO.csv"
    original_xyz = pd.read_csv(original_xyz_path)

    # Select common columns for comparison
    original_xyz = original_xyz.drop("Unnamed: 0", axis=1, errors="ignore")
    common_cols = ["sync_index", "point_id", "x_coord", "y_coord", "z_coord"]
    original_xyz_filtered = original_xyz[common_cols]
    xyz_recalculated_filtered = xyz_recalculated[common_cols]

    # Sort both dataframes by sync_index and point_id to ensure they're aligned
    original_xyz_filtered = original_xyz_filtered.sort_values(["sync_index", "point_id"]).reset_index(drop=True)
    xyz_recalculated_filtered = xyz_recalculated_filtered.sort_values(["sync_index", "point_id"]).reset_index(drop=True)

    # Make sure both filtered dataframes have the same shape
    assert original_xyz_filtered.shape == xyz_recalculated_filtered.shape, (
        f"Shape mismatch: original {original_xyz_filtered.shape}, recalculated {xyz_recalculated_filtered.shape}"
    )

    # Define acceptable tolerance for floating point comparisons
    MAX_DEVIATION_METERS = 0.015

    # Compare coordinates with tolerance
    coord_diff_x = abs(original_xyz_filtered["x_coord"] - xyz_recalculated_filtered["x_coord"])
    coord_diff_y = abs(original_xyz_filtered["y_coord"] - xyz_recalculated_filtered["y_coord"])
    coord_diff_z = abs(original_xyz_filtered["z_coord"] - xyz_recalculated_filtered["z_coord"])

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
    logger.info(f"{pct_close_z:.1f}% of z coordinates are within strict tolerance")

    output_path = tmp_path / "xyz_recalculated_from_files.csv"
    xyz_recalculated = pd.DataFrame(xyz_recalculated)
    xyz_recalculated.to_csv(output_path)


if __name__ == "__main__":
    import caliscope.logger

    caliscope.logger.setup_logging()
    temp = Path(__file__).parent / "tmp"
    temp.mkdir(exist_ok=True)
    test_image_points_to_world_points(temp / "image_points_to_world_points")
    test_triangulate_from_files(temp / "triangulate_from_files")
