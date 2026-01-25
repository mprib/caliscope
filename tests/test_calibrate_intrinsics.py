"""Tests for pure intrinsic calibration functions.

These tests validate:
1. Basic calibration produces valid camera parameters
2. Holdout error computation works correctly
3. Frame selection improves generalization vs naive selection
"""

import logging
from pathlib import Path

import pytest

from caliscope import __root__
from caliscope.core.calibrate_intrinsics import (
    IntrinsicCalibrationResult,
    calibrate_intrinsics,
)
from caliscope.core.point_data import ImagePoints

logger = logging.getLogger(__name__)

# Test data paths
PRERECORDED_SESSION = Path(__root__, "tests", "sessions", "prerecorded_calibration")
INTRINSIC_CSV = PRERECORDED_SESSION / "calibration" / "intrinsic" / "CHARUCO" / "xy_CHARUCO.csv"

# Image size from prerecorded calibration (typical webcam resolution)
IMAGE_SIZE = (1280, 720)


def _load_test_data() -> tuple[ImagePoints, list[int]]:
    """Load test data and return (image_points, port0_frames)."""
    image_points = ImagePoints.from_csv(INTRINSIC_CSV)
    df = image_points.df
    port0_frames = sorted(df[df["port"] == 0]["sync_index"].unique().tolist())
    return image_points, port0_frames


class TestCalibrateIntrinsics:
    """Unit tests for calibrate_intrinsics function."""

    def test_calibrate_returns_valid_result(self):
        """Calibration produces reasonable camera matrix and distortions."""
        image_points, port0_frames = _load_test_data()

        # Use first 20 frames for calibration
        selected = port0_frames[:20]

        result = calibrate_intrinsics(
            image_points,
            port=0,
            image_size=IMAGE_SIZE,
            selected_frames=selected,
        )

        # Check result type
        assert isinstance(result, IntrinsicCalibrationResult)

        # Camera matrix should be 3x3
        assert result.camera_matrix.shape == (3, 3)

        # Focal lengths should be positive and reasonable for the image size
        fx, fy = result.camera_matrix[0, 0], result.camera_matrix[1, 1]
        assert fx > 0
        assert fy > 0
        # Focal length can vary widely depending on lens/sensor (0.3x to 5x image dimension)
        assert 0.3 * IMAGE_SIZE[0] < fx < 5 * IMAGE_SIZE[0]
        assert 0.3 * IMAGE_SIZE[1] < fy < 5 * IMAGE_SIZE[1]

        # Principal point should be near image center
        cx, cy = result.camera_matrix[0, 2], result.camera_matrix[1, 2]
        assert 0.3 * IMAGE_SIZE[0] < cx < 0.7 * IMAGE_SIZE[0]
        assert 0.3 * IMAGE_SIZE[1] < cy < 0.7 * IMAGE_SIZE[1]

        # Standard model has 5 distortion coefficients
        assert result.distortions.shape == (5,)

        # Reprojection error should be reasonable (< 1 pixel is good)
        assert 0 < result.reprojection_error < 2.0

        # Frames used should match selected (or less if some had insufficient corners)
        assert 0 < result.frames_used <= len(selected)

    def test_calibrate_insufficient_frames_raises(self):
        """Calibration with no valid frames raises ValueError."""
        image_points, _ = _load_test_data()

        # Empty frame list
        with pytest.raises(ValueError, match="No valid calibration frames"):
            calibrate_intrinsics(
                image_points,
                port=0,
                image_size=IMAGE_SIZE,
                selected_frames=[],
            )

    def test_calibrate_nonexistent_port_raises(self):
        """Calibration for non-existent port raises ValueError."""
        image_points, _ = _load_test_data()

        with pytest.raises(ValueError, match="No valid calibration frames"):
            calibrate_intrinsics(
                image_points,
                port=999,  # Non-existent port
                image_size=IMAGE_SIZE,
                selected_frames=[0, 1, 2],
            )


# NOTE: Frame selection validation tests are deferred.
#
# The goal is to prove that select_calibration_frames() produces calibrations
# that generalize better than naive frame selection, measured by out-of-sample
# RMSE using compute_holdout_error().
#
# However, the current frame selector has design issues that need to be addressed:
# 1. Coverage-based filtering excludes frames where the board appears small (far away)
# 2. But distance diversity (board at varying distances) is actually valuable for calibration
# 3. The selector should optimize for diversity across multiple dimensions:
#    - Spatial coverage (where in image)
#    - Orientation diversity (board tilt/rotation)
#    - Distance diversity (board size in frame)
#
# See .todo_capture.md for follow-up issue to revisit the selection algorithm
# and create a proper testing framework for intrinsic calibration quality.


class TestCalibrationPersistence:
    """Integration tests for calibration output persistence.

    Tests the full flow: run_intrinsic_calibration → repository save → reload.
    This caught a bug where numpy types in the report couldn't be serialized.
    """

    def test_calibration_output_survives_persistence(self, tmp_path):
        """Calibration output can be saved and loaded via repository."""
        from caliscope.cameras.camera_array import CameraData
        from caliscope.core.calibrate_intrinsics import run_intrinsic_calibration
        from caliscope.core.frame_selector import select_calibration_frames
        from caliscope.repositories.intrinsic_report_repository import (
            IntrinsicReportRepository,
        )

        image_points, port0_frames = _load_test_data()

        # Create a minimal CameraData for the calibration
        camera = CameraData(port=0, size=IMAGE_SIZE, rotation_count=0)

        # Run frame selection (produces numpy types internally)
        selection_result = select_calibration_frames(image_points, port=0, image_size=IMAGE_SIZE)

        # Run calibration orchestrator - this produces IntrinsicCalibrationOutput
        output = run_intrinsic_calibration(camera, image_points, selection_result)

        # Verify we got a valid output
        assert output.camera.matrix is not None
        assert output.report.rmse > 0
        assert len(output.report.selected_frames) > 0

        # Save via repository (this is where numpy types caused problems)
        repo = IntrinsicReportRepository(tmp_path / "reports")
        repo.save(port=0, report=output.report)

        # Load back
        loaded_report = repo.load(port=0)

        # Verify round-trip preserves values
        assert loaded_report is not None
        assert loaded_report.rmse == pytest.approx(output.report.rmse)
        assert loaded_report.frames_used == output.report.frames_used
        assert loaded_report.coverage_fraction == pytest.approx(output.report.coverage_fraction)
        assert loaded_report.selected_frames == output.report.selected_frames


if __name__ == "__main__":
    import caliscope.logger
    import tempfile

    caliscope.logger.setup_logging()

    # Run actual tests so assertions can be caught in debugger
    test_calibrate = TestCalibrateIntrinsics()
    test_calibrate.test_calibrate_returns_valid_result()
    logger.info("PASS: test_calibrate_returns_valid_result")

    test_calibrate.test_calibrate_insufficient_frames_raises()
    logger.info("PASS: test_calibrate_insufficient_frames_raises")

    test_calibrate.test_calibrate_nonexistent_port_raises()
    logger.info("PASS: test_calibrate_nonexistent_port_raises")

    test_persist = TestCalibrationPersistence()
    with tempfile.TemporaryDirectory() as tmp:
        test_persist.test_calibration_output_survives_persistence(Path(tmp))
    logger.info("PASS: test_calibration_output_survives_persistence")

    logger.info("All tests passed!")
