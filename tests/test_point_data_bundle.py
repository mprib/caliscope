import logging
from pathlib import Path

import numpy as np
import pandas as pd

from caliscope import __root__
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.point_data_bundle import PointDataBundle, BundleMetadata
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.managers.point_data_bundle_manager import PointDataBundleManager
from caliscope.post_processing.point_data import ImagePoints, WorldPoints
from caliscope import persistence

logger = logging.getLogger(__name__)

# Tolerance for RMSE comparison between implementations (pixels)
# Can be adjusted based on numerical precision requirements
RMSE_TOLERANCE = 1e-3


def test_world_data_point_estimates(tmp_path: Path):
    """Test WorldPoints <-> PointEstimates round-trip conversion preserves 3D data integrity."""
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

    # Convert WorldPoints -> PointEstimates -> WorldPoints
    point_estimates_from_world_points = world_points_triangulated.to_point_estimates()
    world_points_from_point_estimates = WorldPoints.from_point_estimates(point_estimates_from_world_points)

    # =========================================================================
    # VALIDATION: PointEstimates structure integrity
    # =========================================================================
    logger.info("=" * 50)
    logger.info("VALIDATING POINTESTIMATES STRUCTURE")
    logger.info("=" * 50)

    # Check 1: Array length consistency
    n_observations = len(point_estimates_from_world_points.sync_indices)
    assert len(point_estimates_from_world_points.camera_indices) == n_observations, "Camera indices length mismatch"
    assert len(point_estimates_from_world_points.point_id) == n_observations, "Point ID length mismatch"
    assert len(point_estimates_from_world_points.img) == n_observations, "Image points length mismatch"
    assert len(point_estimates_from_world_points.obj_indices) == n_observations, "Object indices length mismatch"
    logger.info(f"✓ All arrays have consistent length: {n_observations} observations")

    # Check 2: No orphaned 3D points (critical for bundle adjustment)
    unique_obj_indices = np.unique(point_estimates_from_world_points.obj_indices)
    assert unique_obj_indices.size == point_estimates_from_world_points.obj.shape[0], (
        "CRITICAL: Orphaned 3D points detected! This will cause bundle adjustment to hang."
    )
    logger.info(f"✓ No orphaned 3D points: {point_estimates_from_world_points.obj.shape[0]} unique points")

    # Check 3: Valid camera indices
    unique_camera_indices = np.unique(point_estimates_from_world_points.camera_indices)
    n_posed_cams = len(camera_array.posed_cameras)
    assert unique_camera_indices.max() < n_posed_cams, (
        f"Camera index {unique_camera_indices.max()} out of bounds (max should be {n_posed_cams - 1})"
    )
    logger.info(f"✓ Valid camera indices: {unique_camera_indices.size} cameras")

    # Check 4: Object indices within bounds
    max_obj_index = point_estimates_from_world_points.obj_indices.max()
    n_obj_points = point_estimates_from_world_points.obj.shape[0]
    assert max_obj_index < n_obj_points, (
        f"Object index {max_obj_index} out of bounds (max should be {n_obj_points - 1})"
    )
    logger.info("✓ All object indices within valid bounds")

    # Check 5: Camera count matches posed cameras
    assert point_estimates_from_world_points.n_cameras == n_posed_cams, (
        f"Camera count mismatch: PointEstimates has {point_estimates_from_world_points.n_cameras}, "
        f"but CameraArray has {n_posed_cams} posed cameras"
    )
    logger.info(f"✓ Camera count matches posed cameras: {n_posed_cams}")

    # Check 5b: Image point count matches observations
    assert point_estimates_from_world_points.n_img_points == n_observations, (
        f"Image point count mismatch: n_img_points={point_estimates_from_world_points.n_img_points}, "
        f"but actual observations={n_observations}"
    )
    logger.info(f"✓ Image point count matches observations: {n_observations}")

    # =========================================================================
    # VALIDATION: 3D point data preservation
    # =========================================================================
    logger.info("=" * 50)
    logger.info("VALIDATING 3D POINT DATA PRESERVATION")
    logger.info("=" * 50)

    # Get unique points from original WorldPoints (one per point_id)
    original_unique = world_points_triangulated.df.drop_duplicates(subset="point_id", keep="first")
    original_unique = original_unique.sort_values("point_id").reset_index(drop=True)

    # Get points from reconstructed WorldPoints
    reconstructed = world_points_from_point_estimates.df.sort_values("point_id").reset_index(drop=True)

    # Check 6: Same number of unique points
    assert len(original_unique) == len(reconstructed), (
        f"Point count mismatch: original had {len(original_unique)}, reconstructed has {len(reconstructed)}"
    )
    logger.info(f"✓ Same number of unique points: {len(reconstructed)}")

    # Check 7: Point IDs match exactly
    original_point_ids = original_unique["point_id"].values
    reconstructed_point_ids = reconstructed["point_id"].values
    assert np.array_equal(original_point_ids, reconstructed_point_ids), (
        "Point IDs do not match between original and reconstructed"
    )
    logger.info("✓ Point IDs match exactly")

    # Check 8: Coordinates match within tolerance
    coord_tolerance = 1e-6  # micrometer precision at meter scale
    for axis in ["x_coord", "y_coord", "z_coord"]:
        original_coords = original_unique[axis].values
        reconstructed_coords = reconstructed[axis].values
        max_diff = np.max(np.abs(original_coords - reconstructed_coords))
        assert max_diff < coord_tolerance, (
            f"{axis} coordinate mismatch: max difference {max_diff} exceeds tolerance {coord_tolerance}"
        )
        logger.info(f"✓ {axis} coordinates match within tolerance (max diff: {max_diff:.2e})")

    # Check 9: Sync indices are reasonable (should be first occurrence)
    for point_id in reconstructed["point_id"]:
        original_sync = original_unique.loc[original_unique["point_id"] == point_id, "sync_index"].iloc[0]
        reconstructed_sync = reconstructed.loc[reconstructed["point_id"] == point_id, "sync_index"].iloc[0]
        assert original_sync == reconstructed_sync, f"""
            Sync index mismatch for point_id {point_id}:
            original {original_sync} vs reconstructed {reconstructed_sync}
            """
    logger.info("✓ Sync indices match first occurrence in original data")

    # Check 10: Verify no duplicate point_ids in reconstructed
    assert reconstructed["point_id"].nunique() == len(reconstructed), (
        "Reconstructed WorldPoints contains duplicate point IDs"
    )
    logger.info("✓ No duplicate point IDs in reconstructed data")

    logger.info("=" * 50)
    logger.info("ALL VALIDATION CHECKS PASSED!")
    logger.info("WorldPoints <-> PointEstimates round-trip conversion is working correctly.")
    logger.info("=" * 50)


def test_point_data_bundle(tmp_path: Path):
    """Test PointDataBundle implementation against CaptureVolume workflow."""
    version = "post_optimization"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    # Load data
    logger.info("Loading camera array and charuco...")
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    charuco = persistence.load_charuco(tmp_path / "charuco.toml")

    # Load existing OPTIMIZED point estimates (from bundle adjustment)
    point_estimates = persistence.load_point_estimates(tmp_path / "point_estimates.toml")

    recording_path = Path(tmp_path, "calibration", "extrinsic")
    xy_data_path = Path(recording_path, "CHARUCO", "xy_CHARUCO.csv")
    image_points = ImagePoints.from_csv(xy_data_path)

    logger.info(f"Loaded {len(image_points.df)} image observations from {len(camera_array.cameras)} cameras")

    # Triangulate to create world points
    logger.info("Triangulating image points...")
    # world_points = WorldPoints.from_point_estimates(point_estimates)
    world_points = image_points.triangulate(camera_array)

    logger.info(f"Created {len(world_points.df)} world points")

    # Create bundle metadata (sparse)
    metadata = BundleMetadata(
        created_at=pd.Timestamp.now().isoformat(),
        generation_method="triangulation",
        generation_params={"source_fixture": version, "charuco": str(charuco)},
        camera_array_path=Path("camera_array.toml"),
    )

    # Create PointDataBundle
    logger.info("Creating PointDataBundle...")
    bundle = PointDataBundle(
        camera_array=camera_array,
        image_points=image_points,
        world_points=world_points,
        metadata=metadata,
    )

    # Add this debug block
    logger.info("=" * 50)
    logger.info("DEBUG: Data Structure Analysis")
    logger.info("=" * 50)
    logger.info(f"Image points columns: {list(image_points.df.columns)}")
    logger.info(f"World points columns: {list(world_points.df.columns)}")
    logger.info("PointEstimates structure:")
    logger.info(f"  - sync_indices: {point_estimates.sync_indices.shape}")
    logger.info(f"  - camera_indices: {point_estimates.camera_indices.shape}")
    logger.info(f"  - point_id: {point_estimates.point_id.shape}")
    logger.info(f"  - obj_indices: {point_estimates.obj_indices.shape}")
    logger.info(f"  - obj: {point_estimates.obj.shape}")

    # Check for mismatched identifiers
    logger.info("\nIdentifier Overlap Analysis:")
    img_keys = set(zip(image_points.df["sync_index"], image_points.df["point_id"]))
    world_keys = set(zip(world_points.df["sync_index"], world_points.df["point_id"]))
    pe_keys = set(zip(point_estimates.sync_indices, point_estimates.point_id))

    logger.info(f"Image points unique keys: {len(img_keys)}")
    logger.info(f"World points unique keys: {len(world_keys)}")
    logger.info(f"PointEstimates unique keys: {len(pe_keys)}")
    logger.info(f"World vs Image overlap: {len(world_keys & img_keys)}")
    logger.info(f"World vs PE overlap: {len(world_keys & pe_keys)}")

    # Sample data comparison
    logger.info("\nSample data (first 5 rows):")
    logger.info(f"Image points:\n{image_points.df[['sync_index', 'point_id', 'port']].head()}")
    logger.info(f"World points:\n{world_points.df[['sync_index', 'point_id']].head()}")
    logger.info(
        f"""
        PointEstimates:\n{
            pd.DataFrame(
                {
                    "sync_index": point_estimates.sync_indices[:5],
                    "point_id": point_estimates.point_id[:5],
                    "obj_idx": point_estimates.obj_indices[:5],
                }
            )
        }
        """
    )

    # Test 1: RMSE calculation matches CaptureVolume
    logger.info("=" * 50)
    logger.info("TEST 1: RMSE Calculation")
    logger.info("=" * 50)
    bundle_rmse = bundle.calculate_reprojection_error(normalized=False)

    # Load existing point_estimates for CaptureVolume comparison
    point_estimates = persistence.load_point_estimates(tmp_path / "point_estimates.toml")
    capture_volume = CaptureVolume(camera_array, point_estimates)
    cv_rmse = capture_volume.rmse["overall"]

    logger.info(f"PointDataBundle RMSE: {bundle_rmse:.6f} pixels")
    logger.info(f"CaptureVolume RMSE: {cv_rmse:.6f} pixels")
    logger.info(f"Difference: {abs(bundle_rmse - cv_rmse):.6f} pixels")

    assert abs(bundle_rmse - cv_rmse) < RMSE_TOLERANCE, f"RMSE mismatch: {bundle_rmse} vs {cv_rmse}"
    logger.info("✓ RMSE calculation matches CaptureVolume")

    # Test 2: point_estimates property
    logger.info("\n" + "=" * 50)
    logger.info("TEST 2: PointEstimates Property")
    logger.info("=" * 50)
    bundle_pe = bundle.point_estimates

    # Basic validation of structure
    assert isinstance(bundle_pe, PointEstimates), f"Expected PointEstimates, got {type(bundle_pe)}"
    assert bundle_pe.n_cameras == len(camera_array.posed_cameras), (
        f"Camera count mismatch: {bundle_pe.n_cameras} vs {len(camera_array.posed_cameras)}"
    )
    assert bundle_pe.n_img_points == len(image_points.df), (
        f"Image point count mismatch: {bundle_pe.n_img_points} vs {len(image_points.df)}"
    )
    assert bundle_pe.n_obj_points == len(world_points.df), (
        f"World point count mismatch: {bundle_pe.n_obj_points} vs {len(world_points.df)}"
    )

    logger.info(
        f"""
            Created PointEstimates with {bundle_pe.n_cameras} cameras,
            {bundle_pe.n_img_points} image points, {bundle_pe.n_obj_points} object points
        """
    )
    logger.info("✓ PointEstimates property structure is valid")

    # Test 3: error_breakdown
    logger.info("\n" + "=" * 50)
    logger.info("TEST 3: Error Breakdown")
    logger.info("=" * 50)
    breakdown = bundle.error_breakdown()

    assert "by_camera" in breakdown, "Missing 'by_camera' in breakdown"
    assert "by_point" in breakdown, "Missing 'by_point' in breakdown"
    assert "by_observation" in breakdown, "Missing 'by_observation' in breakdown"

    # Validate by_camera structure
    by_camera = breakdown["by_camera"]
    assert "port" in by_camera.columns, "Missing 'port' column in by_camera"
    assert "mean_error" in by_camera.columns, "Missing 'mean_error' column in by_camera"
    assert len(by_camera) == len(camera_array.posed_cameras), "Camera count mismatch in breakdown"

    logger.info("Per-camera error statistics:")
    for _, row in by_camera.iterrows():
        logger.info(
            f"""
                Camera {row["port"]}:
                mean={row["mean_error"]:.4f},
                std={row["std_error"]:.4f},
                count={row["observation_count"]}
            """
        )

    # Compare with CaptureVolume per-camera RMSE
    cv_rmse_by_cam = capture_volume.rmse
    logger.info("\nComparison with CaptureVolume RMSE:")
    for _, row in by_camera.iterrows():
        port = row["port"]
        bundle_cam_rmse = row["mean_error"]
        cv_cam_rmse = cv_rmse_by_cam[str(port)]
        diff = abs(bundle_cam_rmse - cv_cam_rmse)
        logger.info(f"  Camera {port}: bundle={bundle_cam_rmse:.4f}, cv={cv_cam_rmse:.4f}, diff={diff:.6f}")
        assert diff < RMSE_TOLERANCE, f"Per-camera RMSE mismatch for camera {port}: {bundle_cam_rmse} vs {cv_cam_rmse}"

    logger.info("✓ Error breakdown structure is valid and matches CaptureVolume")

    # Test 4: filter_worst_fraction
    logger.info("\n" + "=" * 50)
    logger.info("TEST 4: Filter Worst Fraction")
    logger.info("=" * 50)
    original_rmse = bundle.calculate_reprojection_error()
    logger.info(f"Original RMSE: {original_rmse:.6f} pixels")
    logger.info(f"Original observations: {len(bundle.image_points.df)}")

    try:
        filtered_bundle = bundle.filter_worst_fraction(
            fraction=0.05,
            strategy="global",
            min_observations_per_camera=20,
            level="observation",
        )

        filtered_rmse = filtered_bundle.calculate_reprojection_error()
        remaining_observations = len(filtered_bundle.image_points.df)

        logger.info(f"Filtered RMSE: {filtered_rmse:.6f} pixels")
        logger.info(f"Remaining observations: {remaining_observations}")
        logger.info(f"RMSE improvement: {original_rmse - filtered_rmse:.6f} pixels")

        # Verify filtering improved RMSE
        assert filtered_rmse <= original_rmse, f"Filtering increased RMSE: {filtered_rmse} vs {original_rmse}"
        assert remaining_observations < len(bundle.image_points.df), "Filtering did not remove observations"

        # Verify metadata was updated
        assert len(filtered_bundle.metadata.operations) == 1, "Filter operation not recorded in metadata"
        assert filtered_bundle.metadata.operations[0]["type"] == "filter", "Wrong operation type in metadata"

        logger.info("✓ Filtering successfully improved RMSE and updated metadata")

    except Exception as e:
        logger.error(f"Filtering test failed: {e}")
        raise

    # Test 5: Save/load roundtrip
    logger.info("\n" + "=" * 50)
    logger.info("TEST 5: Save/Load Roundtrip")
    logger.info("=" * 50)
    bundle_dir = tmp_path / "test_bundle"
    bundle_dir.mkdir(exist_ok=True)

    manager = PointDataBundleManager(bundle_dir)
    logger.info(f"Saving bundle to {bundle_dir}...")
    manager.save(bundle)

    logger.info("Loading bundle back...")
    loaded_bundle = manager.load()

    # Verify data integrity
    assert len(loaded_bundle.image_points.df) == len(bundle.image_points.df), "Image point count mismatch after load"
    assert len(loaded_bundle.world_points.df) == len(bundle.world_points.df), "World point count mismatch after load"

    # Verify RMSE preserved
    loaded_rmse = loaded_bundle.calculate_reprojection_error()
    logger.info(f"Original RMSE: {bundle_rmse:.6f}")
    logger.info(f"Loaded RMSE: {loaded_rmse:.6f}")
    assert abs(loaded_rmse - bundle_rmse) < RMSE_TOLERANCE, (
        f"RMSE changed after save/load: {loaded_rmse} vs {bundle_rmse}"
    )

    # Verify metadata preserved
    assert loaded_bundle.metadata.created_at == bundle.metadata.created_at, "Created at timestamp changed"
    assert loaded_bundle.metadata.generation_method == bundle.metadata.generation_method, "Generation method changed"
    assert len(loaded_bundle.metadata.operations) == len(bundle.metadata.operations), "Operations count changed"

    logger.info("✓ Save/load roundtrip preserved all data and metadata")

    logger.info("\n" + "=" * 50)
    logger.info("ALL TESTS PASSED!")
    logger.info("=" * 50)


if __name__ == "__main__":
    from caliscope.logger import setup_logging

    setup_logging()

    # Create debug directory
    debug_dir = Path(__file__).parent / "debug"
    debug_dir.mkdir(exist_ok=True)

    # Run test
    test_point_data_bundle(debug_dir)
    # test_world_data_point_estimates(debug_dir)
