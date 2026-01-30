import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from caliscope import __root__
from caliscope.core.point_data_bundle import PointDataBundle
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.repositories import PointDataBundleRepository
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope import persistence

logger = logging.getLogger(__name__)

# Tolerance for RMSE comparison after save/load roundtrip
RMSE_TOLERANCE = 1e-6


def test_triangulation_consistency(tmp_path: Path):
    """Test that triangulation produces consistent 3D data."""
    version = "post_optimization"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    # Load data
    logger.info("Loading camera array...")
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")

    recording_path = Path(tmp_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")
    image_points = ImagePoints.from_csv(xy_data_path)
    world_points_triangulated = image_points.triangulate(camera_array)

    # =========================================================================
    # VALIDATION: 3D point data structure
    # =========================================================================
    logger.info("=" * 50)
    logger.info("VALIDATING TRIANGULATION OUTPUT")
    logger.info("=" * 50)

    # Check 1: Basic structure
    n_world_points = len(world_points_triangulated.df)
    assert n_world_points > 0, "No world points produced"
    logger.info(f"✓ Produced {n_world_points} world points")

    # Check 2: All required columns present
    required_cols = ["sync_index", "point_id", "x_coord", "y_coord", "z_coord"]
    for col in required_cols:
        assert col in world_points_triangulated.df.columns, f"Missing required column: {col}"
    logger.info("✓ All required columns present")

    # Check 3: No NaN coordinates
    for axis in ["x_coord", "y_coord", "z_coord"]:
        assert not world_points_triangulated.df[axis].isna().any(), f"NaN values in {axis}"
    logger.info("✓ No NaN coordinates")

    # Check 4: Points are within reasonable range (calibration data should be ~1m scale)
    for axis in ["x_coord", "y_coord", "z_coord"]:
        coords = world_points_triangulated.df[axis]
        assert coords.abs().max() < 10.0, f"{axis} values out of expected range"
    logger.info("✓ Coordinates within expected range")

    logger.info("=" * 50)
    logger.info("ALL VALIDATION CHECKS PASSED!")
    logger.info("=" * 50)


def test_point_data_bundle(tmp_path: Path):
    """Test PointDataBundle implementation with optimized session data."""
    version = "post_optimization"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    # Load data
    logger.info("Loading camera array and charuco...")
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    persistence.load_charuco(tmp_path / "charuco.toml")

    # Load from CSV format (the canonical storage format)
    csv_dir = tmp_path / "calibration" / "extrinsic" / "CHARUCO"
    image_points = ImagePoints.from_csv(csv_dir / "xy_CHARUCO.csv")
    world_points = WorldPoints.from_csv(csv_dir / "xyz_CHARUCO.csv")

    logger.info(f"Loaded {len(image_points.df)} image observations from CSV")
    logger.info(f"Loaded {len(world_points.df)} world points from CSV")

    # Create PointDataBundle
    logger.info("Creating PointDataBundle...")
    bundle = PointDataBundle(
        camera_array=camera_array,
        image_points=image_points,
        world_points=world_points,
    )

    # Test 1: RMSE calculation
    logger.info("=" * 50)
    logger.info("TEST 1: RMSE Calculation")
    logger.info("=" * 50)
    error_report = bundle.reprojection_report
    bundle_rmse = error_report.overall_rmse

    logger.info(f"PointDataBundle RMSE: {bundle_rmse:.6f} pixels")
    assert bundle_rmse > 0, "RMSE should be positive"

    # Verify per-camera RMSE is available
    per_camera_rmse = error_report.by_camera
    assert len(per_camera_rmse) == len(camera_array.posed_cameras)
    logger.info("✓ RMSE calculation complete with per-camera breakdown")

    # Test 2: Save/load roundtrip
    logger.info("\n" + "=" * 50)
    logger.info("TEST 2: Save/Load Roundtrip")
    logger.info("=" * 50)
    bundle_dir = tmp_path / "test_bundle"
    bundle_dir.mkdir(exist_ok=True)

    repository = PointDataBundleRepository(bundle_dir)
    logger.info(f"Saving bundle to {bundle_dir}...")
    repository.save(bundle)

    logger.info("Loading bundle back...")
    loaded_bundle = repository.load()

    # Verify data integrity
    assert len(loaded_bundle.image_points.df) == len(bundle.image_points.df), "Image point count mismatch after load"
    assert len(loaded_bundle.world_points.df) == len(bundle.world_points.df), "World point count mismatch after load"

    # Verify RMSE preserved
    loaded_bundle_report = loaded_bundle.reprojection_report

    loaded_rmse = loaded_bundle_report.overall_rmse

    logger.info(f"Original RMSE: {bundle_rmse:.6f}")
    logger.info(f"Loaded RMSE: {loaded_rmse:.6f}")
    assert abs(loaded_rmse - bundle_rmse) < RMSE_TOLERANCE, (
        f"RMSE changed after save/load: {loaded_rmse} vs {bundle_rmse}"
    )

    logger.info("\n" + "=" * 50)
    logger.info("ALL TESTS PASSED!")
    logger.info("=" * 50)


def test_align_bundle_to_charuco_board(tmp_path: Path):
    """Test aligning a PointDataBundle to Charuco board coordinates."""
    # Setup: load a calibration session with Charuco data including obj_loc coordinates
    # Use larger_calibration_post_monocal which has populated obj_loc values
    version = "larger_calibration_post_monocal"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    # Load data
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    persistence.load_charuco(tmp_path / "charuco.toml")
    image_points = ImagePoints.from_csv(tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv")

    # This session has obj_loc_x and obj_loc_y populated, but obj_loc_z may be missing (planar board)
    # Set obj_loc_z to 0.0 for planar board assumption
    df = image_points.df.copy()
    if df["obj_loc_z"].isna().all():
        logger.info("obj_loc_z is all NaN, assuming planar board with z=0")
        df["obj_loc_z"] = 0.0
    image_points = ImagePoints(df)

    # Verify we have valid object coordinates
    assert not image_points.df["obj_loc_x"].isna().any(), "obj_loc_x contains NaN values"
    assert not image_points.df["obj_loc_z"].isna().any(), "obj_loc_z contains NaN values"

    # Create initial bundle (in arbitrary reconstruction units)
    world_points = image_points.triangulate(camera_array)
    bundle = PointDataBundle(camera_array, image_points, world_points)

    # Select a sync_index where board is well-visible (most detections)
    sync_index_counts = image_points.df["sync_index"].value_counts()
    sync_index = int(sync_index_counts.idxmax())  # idxmax returns int | str, but sync_index is always int
    logger.info(f"Using sync_index {sync_index} for alignment (has {sync_index_counts.max()} detections)")

    # Align to object
    aligned_bundle = bundle.align_to_object(sync_index)

    # Verification 1: RMSE should be preserved (geometrically identical)
    original_rmse = bundle.reprojection_report.overall_rmse
    aligned_rmse = aligned_bundle.reprojection_report.overall_rmse
    assert abs(original_rmse - aligned_rmse) < 1e-6, f"RMSE changed after alignment: {original_rmse} vs {aligned_rmse}"

    # Verification 2: Retriangulate with aligned cameras
    # The retriangulated points should match the aligned world points
    retriangulated_points = aligned_bundle.image_points.triangulate(aligned_bundle.camera_array)

    # Compare retriangulated vs aligned world points
    merged_points = pd.merge(
        retriangulated_points.df,
        aligned_bundle.world_points.df,
        on=["sync_index", "point_id"],
        suffixes=("_retri", "_aligned"),
    )

    # They should be very close (numerical differences only)
    for axis in ["x", "y", "z"]:
        diff = np.abs(merged_points[f"{axis}_coord_retri"] - merged_points[f"{axis}_coord_aligned"])
        max_diff = diff.max()
        # 1e-4 -> 0.1mm precision
        assert max_diff < 1e-4, f"Retriangulated points don't match aligned points for {axis} axis: max diff {max_diff}"

    # Verification 3: At alignment sync_index, points should match object coordinates
    aligned_world_at_sync = aligned_bundle.world_points.df[aligned_bundle.world_points.df["sync_index"] == sync_index]

    # Get object coordinates from image points
    img_at_sync = image_points.df[image_points.df["sync_index"] == sync_index]
    merged_at_sync = pd.merge(
        aligned_world_at_sync, img_at_sync[["point_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]], on="point_id"
    )

    # Check that aligned points match object coordinates (within tolerance for noise)
    for axis in ["x", "y", "z"]:
        coord_diff = np.abs(merged_at_sync[f"{axis}_coord"] - merged_at_sync[f"obj_loc_{axis}"])
        max_diff = coord_diff.max()
        # Allow 1cm tolerance for noise/reprojection errors
        assert max_diff < 0.01, (
            f"Aligned points don't match object coordinates for {axis} axis at sync_index {sync_index}: "
            f"max diff {max_diff}"
        )

    logger.info("✓ All alignment validations passed")


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_rotation_invariance(axis: Literal["x", "y", "z"], tmp_path: Path):
    """
    Tests that 4x90-degree rotations around any axis returns bundle to original state.

    This is the migrated version of test_rotation_invariance from test_capture_volume_transformation.py.
    Uses PointDataBundle's immutable rotation API instead of CaptureVolume's mutable rotate() method.
    """
    # SETUP: Use optimized session with stable calibration
    source_session_path = Path(__root__, "tests", "sessions", "post_optimization")
    copy_contents_to_clean_dest(source_session_path, tmp_path)

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")

    # Load from CSV format
    csv_dir = tmp_path / "calibration" / "extrinsic" / "CHARUCO"
    image_points = ImagePoints.from_csv(csv_dir / "xy_CHARUCO.csv")
    world_points = WorldPoints.from_csv(csv_dir / "xyz_CHARUCO.csv")
    bundle = PointDataBundle(camera_array, image_points, world_points)

    # STORE INITIAL STATE
    initial_points = bundle.world_points.points.copy()
    initial_transforms = {port: cam.transformation.copy() for port, cam in bundle.camera_array.posed_cameras.items()}

    logger.info(f"Testing rotation invariance around {axis} axis")
    logger.info(f"Initial state: {len(initial_points)} points, {len(initial_transforms)} cameras")

    # EXECUTE & ASSERT: 4x90-degree rotations
    current_bundle = bundle
    for i in range(1, 5):
        logger.info(f"Applying rotation {i}/4 ({i * 90} degrees total)")

        # Rotate 90 degrees (immutable operation returns new bundle)
        current_bundle = current_bundle.rotate(axis, 90.0)

        current_points = current_bundle.world_points.points
        current_transforms = {
            port: cam.transformation for port, cam in current_bundle.camera_array.posed_cameras.items()
        }

        if i < 4:
            # After 90, 180, 270 degrees: state should be DIFFERENT
            assert not np.allclose(initial_points, current_points, atol=1e-6), (
                f"Points should not match initial state after {i * 90} degrees"
            )

            for port in initial_transforms:
                assert not np.allclose(initial_transforms[port], current_transforms[port], atol=1e-6), (
                    f"Camera {port} transform should not match initial state after {i * 90} degrees"
                )

            logger.info(f"  ✓ State is different after {i * 90} degrees (as expected)")
        else:
            # After 360 degrees: state should RETURN to original
            points_match = np.allclose(initial_points, current_points, atol=1e-6)
            assert points_match, (
                f"Points should return to initial state after 360 degrees\n"
                f"Max difference: {np.max(np.abs(initial_points - current_points))}"
            )

            for port in initial_transforms:
                transform_match = np.allclose(initial_transforms[port], current_transforms[port], atol=1e-6)
                assert transform_match, (
                    f"Camera {port} transform should return to initial state after 360 degrees\n"
                    f"Max difference: {np.max(np.abs(initial_transforms[port] - current_transforms[port]))}"
                )

            logger.info("  ✓ State returned to initial after 360 degrees (rotation invariance confirmed)")

    logger.info(f"✓ Rotation invariance test passed for {axis} axis")


def test_bundle_filter(tmp_path: Path):
    """Test filtering workflow with PointDataBundle.

    Moved from test_optimization_unlinked.py during test consolidation.
    """
    # SETUP: Use post_optimization session with enough data for filtering
    version = "post_optimization"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")

    # Load from CSV format
    csv_dir = tmp_path / "calibration" / "extrinsic" / "CHARUCO"
    image_points = ImagePoints.from_csv(csv_dir / "xy_CHARUCO.csv")
    world_points = WorldPoints.from_csv(csv_dir / "xyz_CHARUCO.csv")

    logger.info("Creating PointDataBundle from loaded data")
    bundle = PointDataBundle(camera_array, image_points, world_points)
    logger.info("PointDataBundle initialized")

    logger.info("Point counts BEFORE filtering:")
    logger.info(f"  3D points: {len(bundle.world_points.df)}")
    logger.info(f"  2D observations: {len(bundle.image_points.df)}")
    logger.info(f"  Cameras: {len(bundle.camera_array.posed_cameras)}")

    # Save initial state
    initial_repo = PointDataBundleRepository(tmp_path / "initial")
    initial_repo.save(bundle)

    # Optimize
    optimized_bundle = bundle.optimize()
    optimized_repo = PointDataBundleRepository(tmp_path / "post_optimization")
    optimized_repo.save(optimized_bundle)

    # Filter out worst 50% of points (percentile filtering)
    filtered_percentile = 50  # Keep best 50%
    logger.info(f"Filtering to keep best {filtered_percentile}% of points")
    filtered_bundle = optimized_bundle.filter_by_percentile_error(
        percentile=filtered_percentile, scope="per_camera", min_per_camera=10
    )
    filtered_repo = PointDataBundleRepository(tmp_path / "post_filtering")
    filtered_repo.save(filtered_bundle)

    logger.info("Point counts AFTER filtering:")
    logger.info(f"  3D points: {len(filtered_bundle.world_points.df)}")
    logger.info(f"  2D observations: {len(filtered_bundle.image_points.df)}")
    logger.info(f"  Cameras: {len(filtered_bundle.camera_array.posed_cameras)}")

    # Re-optimize with filtered data
    reoptimized_bundle = filtered_bundle.optimize()
    reopt_repo = PointDataBundleRepository(tmp_path / "post_filtering_then_optimizing")
    reopt_repo.save(reoptimized_bundle)

    # Verify RMSE improves through the filtering and re-optimization stages
    initial_rmse = bundle.reprojection_report.overall_rmse
    optimized_rmse = optimized_bundle.reprojection_report.overall_rmse
    filtered_rmse = filtered_bundle.reprojection_report.overall_rmse
    final_rmse = reoptimized_bundle.reprojection_report.overall_rmse

    logger.info(f"RMSE progression: {initial_rmse:.4f} → {optimized_rmse:.4f} → {filtered_rmse:.4f} → {final_rmse:.4f}")

    # Note: Initial optimization may not always improve RMSE when starting from
    # ground truth data, but filtering worst observations should always help
    assert filtered_rmse <= optimized_rmse, "Filtering should improve RMSE"
    assert final_rmse <= filtered_rmse, "Second optimization should improve or maintain RMSE"


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()

    # Create debug directory
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)

    # Run test
    test_point_data_bundle(debug_dir)
    test_triangulation_consistency(debug_dir)
    test_align_bundle_to_charuco_board(debug_dir)
