"""Tests for pure intrinsic calibration functions.

These tests validate:
1. Basic calibration produces valid camera parameters
2. Holdout error computation works correctly
3. Frame selection improves generalization vs naive selection
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from caliscope import __root__
from caliscope.core.calibrate_intrinsics import (
    HoldoutResult,
    IntrinsicCalibrationResult,
    calibrate_intrinsics,
    compute_holdout_error,
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


class TestComputeHoldoutError:
    """Unit tests for compute_holdout_error function."""

    def test_holdout_error_returns_valid_result(self):
        """Holdout error computation produces reasonable results."""
        image_points, port0_frames = _load_test_data()

        # Calibrate on first 30 frames
        train_frames = port0_frames[:30]
        calibration_result = calibrate_intrinsics(
            image_points,
            port=0,
            image_size=IMAGE_SIZE,
            selected_frames=train_frames,
        )

        holdout_frames = port0_frames[30:]  # Use frames not in training

        result = compute_holdout_error(
            image_points,
            calibration_result,
            port=0,
            holdout_frames=holdout_frames,
        )

        assert isinstance(result, HoldoutResult)

        # RMSE should be a positive number (not NaN)
        assert not np.isnan(result.rmse)
        assert result.rmse > 0

        # RMSE in pixels should be reasonable (< 5 pixels is good)
        assert not np.isnan(result.rmse_pixels)
        assert result.rmse_pixels < 5.0

        # Per-frame RMSE should have entries for successful frames
        assert len(result.per_frame_rmse) > 0

        # Total frames should match input
        assert result.total_frames == len(holdout_frames)

        # Should have evaluated some points
        assert result.total_points > 0

    def test_holdout_empty_frames_returns_nan(self):
        """Holdout with no valid frames returns NaN RMSE."""
        image_points, port0_frames = _load_test_data()

        # Calibrate first
        train_frames = port0_frames[:30]
        calibration_result = calibrate_intrinsics(
            image_points,
            port=0,
            image_size=IMAGE_SIZE,
            selected_frames=train_frames,
        )

        result = compute_holdout_error(
            image_points,
            calibration_result,
            port=0,
            holdout_frames=[],
        )

        assert np.isnan(result.rmse)
        assert result.total_frames == 0
        assert result.total_points == 0


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


if __name__ == "__main__":
    import caliscope.logger

    caliscope.logger.setup_logging()

    # Run actual tests so assertions can be caught in debugger
    test_calibrate = TestCalibrateIntrinsics()
    test_calibrate.test_calibrate_returns_valid_result()
    logger.info("PASS: test_calibrate_returns_valid_result")

    test_calibrate.test_calibrate_insufficient_frames_raises()
    logger.info("PASS: test_calibrate_insufficient_frames_raises")

    test_calibrate.test_calibrate_nonexistent_port_raises()
    logger.info("PASS: test_calibrate_nonexistent_port_raises")

    test_holdout = TestComputeHoldoutError()
    test_holdout.test_holdout_error_returns_valid_result()
    logger.info("PASS: test_holdout_error_returns_valid_result")

    test_holdout.test_holdout_empty_frames_returns_nan()
    logger.info("PASS: test_holdout_empty_frames_returns_nan")

    logger.info("All tests passed!")
