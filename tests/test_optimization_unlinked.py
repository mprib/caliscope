import logging
from pathlib import Path

from caliscope import __root__
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.core.point_data import ImagePoints
from caliscope.core.point_data_bundle import PointDataBundle
from caliscope import persistence

logger = logging.getLogger(__name__)


def test_bundle_adjust_with_unlinked_camera(tmp_path: Path):
    """
    Tests the full pipeline from initializing a CameraArray with a missing
    camera, through world point generation, to bundle adjustment.
    This ensures that the bundle correctly handles cameras that have no
    shared observations and filters them appropriately.
    """
    # 1. SETUP: Use test data that results in an unposed camera
    version = "not_sufficient_stereopairs"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    # The xy_CHARUCO.csv is at the root of the session for this test case
    xy_data_path = Path(tmp_path, "xy_CHARUCO.csv")

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")

    logger.info("Creating stereocalibrator")
    image_points = ImagePoints.from_csv(xy_data_path)

    # note: using stereocalibrate to attempt to improve speed of calibration
    paired_pose_network: PairedPoseNetwork = build_paired_pose_network(
        image_points, camera_array, method="stereocalibrate"
    )
    logger.info("Initializing estimated camera positions based on best daisy-chained stereopairs")
    paired_pose_network.apply_to(camera_array)

    world_points = image_points.triangulate(camera_array)

    # 3. VERIFY SETUP
    # Confirm that we have the expected set of posed and unposed cameras
    # (cam 5 no shared images and cam 4 actively ignored)
    assert set(camera_array.posed_cameras.keys()) == {1, 2, 3, 6}
    assert list(camera_array.unposed_cameras.keys()) == [4, 5]
    assert len(camera_array.posed_port_to_index) == 4  # Critical: only 4 cameras are indexed for optimization

    # 4. CREATE BUNDLE AND OPTIMIZE
    logger.info("Creating PointDataBundle with unlinked camera present...")
    bundle = PointDataBundle(camera_array, image_points, world_points)

    initial_rmse = bundle.reprojection_report.overall_rmse
    logger.info(f"Initial RMSE: {initial_rmse:.4f} pixels")

    # The core of the test: can it optimize without crashing?
    logger.info("Running optimization...")
    optimized_bundle = bundle.optimize()

    # 5. ASSERT SUCCESS
    # If optimize() completes, the test has passed.
    assert optimized_bundle.optimization_status is not None
    assert optimized_bundle.optimization_status.converged

    final_rmse = optimized_bundle.reprojection_report.overall_rmse
    logger.info(f"Final RMSE: {final_rmse:.4f} pixels")
    logger.info("Optimization completed successfully with an unlinked camera present.")


def test_bundle_filter(tmp_path: Path):
    """Test filtering workflow with PointDataBundle."""
    # 1. SETUP: Use post_optimization session with enough data for filtering
    version = "post_optimization"
    original_session_path = Path(__root__, "tests", "sessions", version)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")

    # Load from CSV format
    from caliscope.core.point_data import ImagePoints, WorldPoints

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
    from caliscope.repositories import PointDataBundleRepository

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

    temp = Path(__file__).parent / "debug"
    test_bundle_adjust_with_unlinked_camera(temp)
    test_bundle_filter(temp)
    logger.info("test debug complete")
