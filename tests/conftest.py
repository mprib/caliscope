from dataclasses import dataclass
from pathlib import Path

import pytest

from caliscope import __root__
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.logger import setup_logging
from caliscope import persistence


# Subsampling stride for fast tests. See specs/test-data-reduction.md for analysis.
# stride=20 reduces 141,422 observations to ~7,147 while maintaining 3.1x overdetermination.
FAST_SUBSAMPLE_STRIDE = 20

# Session names as constants to avoid magic strings
LARGER_CALIBRATION_SESSION = "larger_calibration_post_monocal"


@dataclass
class CalibrationTestData:
    """Loaded calibration session data for testing."""

    camera_array: CameraArray
    image_points: ImagePoints
    session_path: Path
    is_subsampled: bool


def _load_calibration_data(
    tmp_path: Path,
    subsample_stride: int | None = None,
) -> CalibrationTestData:
    """Load calibration test data with optional subsampling.

    Args:
        tmp_path: Pytest temp directory for isolation
        subsample_stride: If provided, keep only every Nth sync_index

    Returns:
        CalibrationTestData with loaded camera array and image points
    """
    original_session_path = Path(__root__, "tests", "sessions", LARGER_CALIBRATION_SESSION)
    copy_contents_to_clean_dest(original_session_path, tmp_path)

    xy_data_path = tmp_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
    camera_array = persistence.load_camera_array(tmp_path / "camera_array.toml")
    image_points = ImagePoints.from_csv(xy_data_path)

    is_subsampled = False
    if subsample_stride is not None:
        max_sync_index = image_points.df.sync_index.max()
        keep_indices = set(range(0, max_sync_index + 1, subsample_stride))
        subsampled_df = image_points.df[image_points.df.sync_index.isin(keep_indices)].copy()
        image_points = ImagePoints(subsampled_df)
        is_subsampled = True

    return CalibrationTestData(
        camera_array=camera_array,
        image_points=image_points,
        session_path=tmp_path,
        is_subsampled=is_subsampled,
    )


@pytest.fixture
def larger_calibration_session(tmp_path: Path) -> CalibrationTestData:
    """Load the larger_calibration_post_monocal session with full data.

    Use this for tests that specifically need the full dataset.
    Consider marking such tests with @pytest.mark.slow.
    """
    return _load_calibration_data(tmp_path, subsample_stride=None)


@pytest.fixture
def larger_calibration_session_reduced(tmp_path: Path) -> CalibrationTestData:
    """Load the larger_calibration_post_monocal session with subsampled data.

    Subsamples to every Nth frame (N=FAST_SUBSAMPLE_STRIDE) for ~6x speedup
    while maintaining sufficient observations for bundle adjustment convergence.
    See specs/test-data-reduction.md for analysis.
    """
    return _load_calibration_data(tmp_path, subsample_stride=FAST_SUBSAMPLE_STRIDE)


@pytest.fixture(scope="session", autouse=True)
def setup_app_logging():
    """Configure the application's logging for the entire test session."""
    setup_logging()
