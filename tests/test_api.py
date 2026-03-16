"""Tests for the caliscope scripting API (src/caliscope/api.py)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliscope import __root__
from caliscope.api import (
    CameraArray,
    CameraData,
    CaptureVolume,
    CharucoTracker,
    Charuco,
    ImagePoints,
    IntrinsicCalibrationOutput,
    CalibrationError,
    calibrate_intrinsics,
    extract_image_points,
    extract_image_points_multicam,
)
from caliscope.helper import copy_contents_to_clean_dest

logger = logging.getLogger(__name__)

# Paths to test sessions
PRERECORDED_SESSION = Path(__root__, "tests", "sessions", "prerecorded_calibration")
CHARUCO_SESSION = Path(__root__, "tests", "sessions", "charuco_calibration")


# ---------------------------------------------------------------------------
# Charuco
# ---------------------------------------------------------------------------


def test_charuco_from_squares():
    """from_squares computes board dimensions from column/row counts and square size."""
    charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)

    assert charuco.board_height == 15.0
    assert charuco.board_width == 12.0
    assert charuco.units == "cm"
    assert charuco.square_size_override_cm == 3.0


# ---------------------------------------------------------------------------
# CameraArray
# ---------------------------------------------------------------------------


def test_camera_array_from_image_sizes():
    """from_image_sizes creates uncalibrated cameras; newly created cameras are not posed."""
    sizes = {0: (1920, 1080), 1: (1280, 720)}
    cameras = CameraArray.from_image_sizes(sizes)

    assert cameras[0].size == (1920, 1080)
    assert cameras[1].size == (1280, 720)
    assert len(cameras.posed_cameras) == 0


# ---------------------------------------------------------------------------
# extract_image_points — error paths
# ---------------------------------------------------------------------------


def test_extract_image_points_missing_files():
    """FileNotFoundError must be raised when the video file does not exist."""
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    with pytest.raises(FileNotFoundError) as exc_info:
        extract_image_points("/nonexistent/cam_0.mp4", 0, tracker)

    assert "/nonexistent/cam_0.mp4" in str(exc_info.value)


def test_extract_image_points_frame_step_invalid():
    """frame_step values less than 1 must raise ValueError."""
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    video_path = PRERECORDED_SESSION / "calibration" / "intrinsic" / "cam_0.mp4"

    with pytest.raises(ValueError):
        extract_image_points(video_path, 0, tracker, frame_step=0)

    with pytest.raises(ValueError):
        extract_image_points(video_path, 0, tracker, frame_step=-1)


# ---------------------------------------------------------------------------
# extract_image_points — progress callback and frame_time column
# ---------------------------------------------------------------------------


class _SpyProgressCallback:
    """Records calls to each ProgressCallback method."""

    def __init__(self) -> None:
        self.video_starts: list[tuple[int, int]] = []
        self.frames: list[tuple[int, int, int]] = []
        self.video_completes: list[int] = []

    def on_video_start(self, cam_id: int, total_frames: int) -> None:
        self.video_starts.append((cam_id, total_frames))

    def on_frame(self, cam_id: int, frame_index: int, n_points: int) -> None:
        self.frames.append((cam_id, frame_index, n_points))

    def on_video_complete(self, cam_id: int) -> None:
        self.video_completes.append(cam_id)

    def on_info(self, message: str) -> None:
        pass


def test_extract_image_points_progress_callback():
    """The progress callback must be called once per video and once per frame.

    Also verifies that the returned ImagePoints contain a populated frame_time column.
    """
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    # Use a single camera video to keep the test fast
    video_path = PRERECORDED_SESSION / "calibration" / "intrinsic" / "cam_0.mp4"

    spy = _SpyProgressCallback()
    image_points = extract_image_points(video_path, 0, tracker, progress=spy)

    # on_video_start called once per video
    assert len(spy.video_starts) == 1
    assert spy.video_starts[0][0] == 0  # cam_id matches

    # on_video_complete called once per video
    assert len(spy.video_completes) == 1
    assert spy.video_completes[0] == 0

    # on_frame called once per frame — count must be positive and match total_frames
    reported_frame_count = spy.video_starts[0][1]
    assert reported_frame_count > 0
    # Frames list length should equal the frame count (one call per frame)
    assert len(spy.frames) == reported_frame_count

    # frame_time column must be present and have at least some non-NaN values
    assert "frame_time" in image_points.df.columns
    assert image_points.df["frame_time"].notna().any()


# ---------------------------------------------------------------------------
# extract_image_points — frame_step
# ---------------------------------------------------------------------------


def test_extract_image_points_frame_step():
    """frame_step=5 must produce fewer rows than frame_step=1."""
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    video_path = PRERECORDED_SESSION / "calibration" / "intrinsic" / "cam_0.mp4"

    df_step1 = extract_image_points(video_path, 0, tracker, frame_step=1).df
    df_step5 = extract_image_points(video_path, 0, tracker, frame_step=5).df

    assert len(df_step5) < len(df_step1)


# ---------------------------------------------------------------------------
# calibrate_intrinsics
# ---------------------------------------------------------------------------


def test_calibrate_intrinsics_wrapper():
    """calibrate_intrinsics returns IntrinsicCalibrationOutput with a camera matrix."""
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    video_path = PRERECORDED_SESSION / "calibration" / "intrinsic" / "cam_0.mp4"
    image_points = extract_image_points(video_path, 0, tracker)

    camera = CameraData(cam_id=0, size=(1280, 720))
    output = calibrate_intrinsics(image_points, camera)

    assert isinstance(output, IntrinsicCalibrationOutput)
    assert output.camera.matrix is not None
    assert output.camera.distortions is not None


def test_calibrate_intrinsics_missing_objloc():
    """calibrate_intrinsics raises CalibrationError when obj_loc is all NaN."""
    # Build ImagePoints with all-NaN obj_loc
    df = pd.DataFrame(
        {
            "sync_index": [0, 0],
            "cam_id": [0, 0],
            "point_id": [0, 1],
            "img_loc_x": [100.0, 200.0],
            "img_loc_y": [150.0, 250.0],
            "obj_loc_x": [np.nan, np.nan],
            "obj_loc_y": [np.nan, np.nan],
            "obj_loc_z": [np.nan, np.nan],
        }
    )
    image_points = ImagePoints(df)
    camera = CameraData(cam_id=0, size=(1280, 720))

    with pytest.raises(CalibrationError):
        calibrate_intrinsics(image_points, camera)


# ---------------------------------------------------------------------------
# extract_image_points_multicam
# ---------------------------------------------------------------------------


def test_extract_image_points_multicam_smoke(tmp_path: Path):
    """Multi-camera extraction with timestamps CSV produces valid ImagePoints."""
    copy_contents_to_clean_dest(CHARUCO_SESSION, tmp_path)

    extrinsic_dir = tmp_path / "calibration" / "extrinsic"
    videos = {
        0: extrinsic_dir / "cam_0.mp4",
        1: extrinsic_dir / "cam_1.mp4",
        2: extrinsic_dir / "cam_2.mp4",
        3: extrinsic_dir / "cam_3.mp4",
    }
    timestamps_path = extrinsic_dir / "timestamps.csv"

    charuco = Charuco.from_toml(tmp_path / "charuco.toml")
    tracker = CharucoTracker(charuco)

    result = extract_image_points_multicam(videos, tracker, frame_step=10, timestamps=timestamps_path)

    assert isinstance(result, ImagePoints)
    assert len(result.df) > 0

    # All four cameras should have contributed observations
    cam_ids_present = set(result.df["cam_id"].unique())
    assert len(cam_ids_present) >= 2, f"Expected >=2 cameras, got {cam_ids_present}"

    # The same sync indices must appear for multiple cameras (this is what
    # makes it useful for extrinsic calibration)
    sync_index_counts = result.df.groupby("sync_index")["cam_id"].nunique()
    multi_cam_sync_indices = (sync_index_counts >= 2).sum()
    assert multi_cam_sync_indices > 0, "No sync indices shared by >=2 cameras"

    # Charuco tracker populates obj_loc columns
    assert "obj_loc_x" in result.df.columns
    assert result.df["obj_loc_x"].notna().any()


def test_extract_image_points_multicam_inferred_timestamps(tmp_path: Path):
    """Multi-camera extraction infers timestamps from video metadata when no CSV is given."""
    copy_contents_to_clean_dest(CHARUCO_SESSION, tmp_path)

    extrinsic_dir = tmp_path / "calibration" / "extrinsic"
    videos = {
        0: extrinsic_dir / "cam_0.mp4",
        1: extrinsic_dir / "cam_1.mp4",
        2: extrinsic_dir / "cam_2.mp4",
        3: extrinsic_dir / "cam_3.mp4",
    }

    charuco = Charuco.from_toml(tmp_path / "charuco.toml")
    tracker = CharucoTracker(charuco)

    # No timestamps parameter — inference path
    result = extract_image_points_multicam(videos, tracker, frame_step=10)

    assert isinstance(result, ImagePoints)
    assert len(result.df) > 0

    cam_ids_present = set(result.df["cam_id"].unique())
    assert len(cam_ids_present) >= 2, f"Expected >=2 cameras, got {cam_ids_present}"

    # Sync indices shared by multiple cameras must exist
    sync_index_counts = result.df.groupby("sync_index")["cam_id"].nunique()
    multi_cam_sync_indices = (sync_index_counts >= 2).sum()
    assert multi_cam_sync_indices > 0, "No sync indices shared by >=2 cameras"

    assert "obj_loc_x" in result.df.columns
    assert result.df["obj_loc_x"].notna().any()


def test_extract_image_points_multicam_pipeline(tmp_path: Path):
    """Full API pipeline: multicam extraction -> bootstrap -> optimize.

    Uses the pre-calibrated camera_array.toml from the charuco_calibration
    session (intrinsics already solved) to focus on the multicam extraction
    and extrinsic pipeline.
    """
    copy_contents_to_clean_dest(CHARUCO_SESSION, tmp_path)

    extrinsic_dir = tmp_path / "calibration" / "extrinsic"
    videos = {
        0: extrinsic_dir / "cam_0.mp4",
        1: extrinsic_dir / "cam_1.mp4",
        2: extrinsic_dir / "cam_2.mp4",
        3: extrinsic_dir / "cam_3.mp4",
    }
    timestamps_path = extrinsic_dir / "timestamps.csv"

    charuco = Charuco.from_toml(tmp_path / "charuco.toml")
    tracker = CharucoTracker(charuco)

    # The session ships with a camera_array.toml that has calibrated intrinsics
    # and extrinsics. Load just the intrinsics (extrinsics will be reset by
    # bootstrap).
    cameras = CameraArray.from_toml(tmp_path / "camera_array.toml")

    # Step 1: extract synchronized multicam image points
    image_points = extract_image_points_multicam(videos, tracker, frame_step=10, timestamps=timestamps_path)

    # Step 2: bootstrap (builds pose network, triangulates world points) and optimize
    optimized = CaptureVolume.bootstrap(image_points, cameras).optimize()

    assert optimized.optimization_status is not None
    assert optimized.optimization_status.converged, "Bundle adjustment did not converge"

    rmse = optimized.reprojection_report.overall_rmse
    logger.info(f"Pipeline test RMSE: {rmse:.4f} px")
    assert rmse < 2.0, f"Reprojection RMSE {rmse:.4f} px exceeds 2.0 px threshold"


# ---------------------------------------------------------------------------
# Debug harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).parent))

    from caliscope.logger import setup_logging

    setup_logging()

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    logger.info("test_charuco_from_squares")
    test_charuco_from_squares()

    logger.info("test_camera_array_from_image_sizes")
    test_camera_array_from_image_sizes()

    logger.info("test_extract_image_points_missing_files")
    test_extract_image_points_missing_files()

    logger.info("test_extract_image_points_frame_step_invalid")
    test_extract_image_points_frame_step_invalid()

    logger.info("test_extract_image_points_progress_callback")
    test_extract_image_points_progress_callback()

    logger.info("test_extract_image_points_frame_step")
    test_extract_image_points_frame_step()

    logger.info("test_calibrate_intrinsics_wrapper")
    test_calibrate_intrinsics_wrapper()

    logger.info("test_calibrate_intrinsics_missing_objloc")
    test_calibrate_intrinsics_missing_objloc()

    logger.info("test_extract_image_points_multicam_smoke")
    with tempfile.TemporaryDirectory() as tmp:
        test_extract_image_points_multicam_smoke(Path(tmp))

    logger.info("test_extract_image_points_multicam_inferred_timestamps")
    with tempfile.TemporaryDirectory() as tmp:
        test_extract_image_points_multicam_inferred_timestamps(Path(tmp))

    logger.info("test_extract_image_points_multicam_pipeline")
    with tempfile.TemporaryDirectory() as tmp:
        test_extract_image_points_multicam_pipeline(Path(tmp))

    logger.info("All API tests passed.")
