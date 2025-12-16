import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend

import logging
from pathlib import Path
import tempfile
import numpy as np

from caliscope import __root__
from caliscope.calibration.point_data_bundle import PointDataBundle
from caliscope.post_processing.point_data import ImagePoints, WorldPoints
from caliscope import persistence
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.logger import setup_logging

logger = logging.getLogger(__name__)


def test_reprojection_report_generation(tmp_path: Path):
    """Integration test: generate report from real post-optimization data."""
    # Setup - copy test session to temp directory
    session_name = "post_optimization"
    original_session_path = Path(__root__, "tests", "sessions", session_name)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    # Load calibration data
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    persistence.load_charuco(tmp_path / "charuco.toml")

    persistence.load_point_estimates(tmp_path / "point_estimates.toml")
    # image_points = ImagePoints.from_point_estimates(point_estimates, camera_array)
    image_points = ImagePoints.from_csv(tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv")
    # world_points = WorldPoints.from_point_estimates(point_estimates)
    world_points = image_points.triangulate(camera_array)

    # Create PointDataBundle
    bundle = PointDataBundle(
        camera_array=camera_array,
        image_points=image_points,
        world_points=world_points,
        metadata={
            "created_at": "test_session",
            "generation_method": "bundle_adjustment",
            "generation_params": {"optimizer": "scipy.least_squares"},
            "camera_array_path": tmp_path / "camera_array.toml",
            "source_files": {
                "image_points": tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv",
                "point_estimates": tmp_path / "point_estimates.toml",
            },
        },
    )

    # Generate reprojection report
    report = bundle.get_reprojection_report()

    # === Validation Assertions ===

    # Basic structure
    assert report.n_observations_total > 0, "Must have observations"
    assert report.n_observations_matched > 0, "Must have matched observations"
    assert report.n_observations_matched <= report.n_observations_total

    # Unmatched tracking
    assert 0 <= report.unmatched_rate <= 1.0
    assert isinstance(report.n_unmatched_observations, int)
    assert len(report.unmatched_by_camera) == len(camera_array.cameras)

    # RMSE values should be reasonable for calibrated system
    assert 0.0 < report.overall_rmse < 10.0, f"RMSE {report.overall_rmse} out of expected range"

    # Per-camera metrics
    for port in camera_array.posed_cameras.keys():
        assert port in report.by_camera, f"Missing RMSE for camera {port}"
        assert 0.0 <= report.by_camera[port] < 10.0

    # Per-point metrics
    assert len(report.by_point_id) > 0, "Should have point-level metrics"
    for point_id, rmse in report.by_point_id.items():
        assert 0.0 <= rmse < 10.0

    # Raw errors DataFrame structure
    assert len(report.raw_errors) == report.n_observations_matched
    expected_columns = ["sync_index", "port", "point_id", "error_x", "error_y", "euclidean_error"]
    assert list(report.raw_errors.columns) == expected_columns

    # Verify error calculations are consistent
    calculated_overall_rmse = float(np.sqrt(np.mean(report.raw_errors["euclidean_error"] ** 2)))
    assert abs(report.overall_rmse - calculated_overall_rmse) < 1e-10

    # Verify caching works (returns same object)
    report2 = bundle.get_reprojection_report()
    assert report2 is report, "Caching failed - should return same object"

    logger.info(
        f"✓ Report generation successful: {report.n_observations_matched} matched, "
        f"{report.n_unmatched_observations} unmatched, "
        f"RMSE = {report.overall_rmse:.4f} pixels"
    )


def test_unmatched_observation_tracking(tmp_path: Path):
    """Verify unmatched observations are correctly tracked."""
    # Use same data as above
    session_name = "post_optimization"
    original_session_path = Path(__root__, "tests", "sessions", session_name)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    image_points = ImagePoints.from_csv(tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv")
    point_estimates = persistence.load_point_estimates(tmp_path / "point_estimates.toml")
    world_points = WorldPoints.from_point_estimates(point_estimates)

    bundle = PointDataBundle(
        camera_array=camera_array,
        image_points=image_points,
        world_points=world_points,
        metadata={"created_at": "test", "generation_method": "test"},
    )

    report = bundle.get_reprojection_report()

    # Verify unmatched counting logic
    total_observations = len(bundle.image_points.df)
    matched_observations = report.n_observations_matched

    # Manual verification of unmatched by camera
    for port in camera_array.cameras.keys():
        port_total = (bundle.image_points.df["port"] == port).sum()
        port_matched = ((bundle.image_points.df["port"] == port) & (bundle.img_to_obj_map >= 0)).sum()
        expected_unmatched = port_total - port_matched

        assert report.unmatched_by_camera[port] == expected_unmatched, f"Unmatched count mismatch for camera {port}"

    assert report.n_unmatched_observations == total_observations - matched_observations
    logger.info("✓ Unmatched observation tracking validated")


if __name__ == "__main__":
    setup_logging()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print("Running reprojection report tests...")
        test_reprojection_report_generation(tmp_path)
        test_unmatched_observation_tracking(tmp_path)
