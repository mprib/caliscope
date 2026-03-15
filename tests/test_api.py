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
)

logger = logging.getLogger(__name__)

# Paths to test sessions
PRERECORDED_SESSION = Path(__root__, "tests", "sessions", "prerecorded_calibration")
POST_OPTIMIZATION_SESSION = Path(__root__, "tests", "sessions", "post_optimization")


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


def test_camera_array_getitem_setitem():
    """CameraArray supports dict-style indexed read and write."""
    cameras = CameraArray.from_image_sizes({0: (1920, 1080), 1: (1280, 720)})

    # __getitem__
    cam0 = cameras[0]
    assert cam0.cam_id == 0
    assert cam0.size == (1920, 1080)

    # __setitem__
    new_cam = CameraData(cam_id=2, size=(640, 480))
    cameras[2] = new_cam
    assert cameras[2].size == (640, 480)


def test_camera_array_from_image_sizes():
    """from_image_sizes creates uncalibrated cameras with the given resolutions."""
    sizes = {0: (1920, 1080), 1: (1280, 720)}
    cameras = CameraArray.from_image_sizes(sizes)

    assert len(cameras.cameras) == 2
    assert cameras[0].size == (1920, 1080)
    assert cameras[1].size == (1280, 720)

    # Newly created cameras should not be posed
    assert len(cameras.posed_cameras) == 0


# ---------------------------------------------------------------------------
# CaptureVolume save / load roundtrip
# ---------------------------------------------------------------------------


def test_capture_volume_save_load_roundtrip(tmp_path: Path):
    """Save a CaptureVolume to disk then reload it; metadata must be preserved."""
    # Build a CaptureVolume from the post_optimization session files
    image_points_path = POST_OPTIMIZATION_SESSION / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
    camera_array_path = POST_OPTIMIZATION_SESSION / "camera_array.toml"
    world_points_path = POST_OPTIMIZATION_SESSION / "calibration" / "extrinsic" / "CHARUCO" / "xyz_CHARUCO.csv"

    from caliscope.core.point_data import WorldPoints

    camera_array = CameraArray.from_toml(camera_array_path)
    image_points = ImagePoints.from_csv(image_points_path)
    world_points = WorldPoints.from_csv(world_points_path)

    original = CaptureVolume(
        camera_array=camera_array,
        image_points=image_points,
        world_points=world_points,
    )

    # Save to tmp_path, then load back
    save_dir = tmp_path / "capture_volume"
    original.save(save_dir)
    reloaded = CaptureVolume.load(save_dir)

    # Camera count and world point count must be preserved
    assert len(reloaded.camera_array.posed_cameras) == len(original.camera_array.posed_cameras)
    assert len(reloaded.world_points.df) == len(original.world_points.df)


# ---------------------------------------------------------------------------
# extract_image_points — error paths
# ---------------------------------------------------------------------------


def test_extract_image_points_missing_files():
    """FileNotFoundError must be raised when video files do not exist."""
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    fake_videos = {
        0: "/nonexistent/cam_0.mp4",
        3: "/nonexistent/cam_3.mp4",
    }

    with pytest.raises(FileNotFoundError) as exc_info:
        extract_image_points(fake_videos, tracker)

    error_message = str(exc_info.value)
    # Both missing cam_ids should be mentioned
    assert "0" in error_message
    assert "3" in error_message


# ---------------------------------------------------------------------------
# extract_image_points — progress callback
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


def test_extract_image_points_progress_callback():
    """The progress callback must be called once per video and once per frame."""
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    # Use a single camera video to keep the test fast
    video_path = PRERECORDED_SESSION / "calibration" / "intrinsic" / "cam_0.mp4"
    videos = {0: video_path}

    spy = _SpyProgressCallback()
    extract_image_points(videos, tracker, progress=spy)

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


# ---------------------------------------------------------------------------
# extract_image_points — frame_time column
# ---------------------------------------------------------------------------


def test_extract_image_points_includes_frame_time():
    """Extracted ImagePoints must contain a non-all-NaN frame_time column."""
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    video_path = PRERECORDED_SESSION / "calibration" / "intrinsic" / "cam_0.mp4"
    videos = {0: video_path}

    image_points = extract_image_points(videos, tracker)

    assert "frame_time" in image_points.df.columns
    # At least some frame_time values must be non-NaN
    assert image_points.df["frame_time"].notna().any()


# ---------------------------------------------------------------------------
# calibrate_intrinsics
# ---------------------------------------------------------------------------


def test_calibrate_intrinsics_wrapper():
    """calibrate_intrinsics returns IntrinsicCalibrationOutput with a camera matrix."""
    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    video_path = PRERECORDED_SESSION / "calibration" / "intrinsic" / "cam_0.mp4"
    image_points = extract_image_points({0: video_path}, tracker)

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
# Debug harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    from caliscope.logger import setup_logging

    setup_logging()

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    logger.info("test_charuco_from_squares")
    test_charuco_from_squares()

    logger.info("test_camera_array_getitem_setitem")
    test_camera_array_getitem_setitem()

    logger.info("test_camera_array_from_image_sizes")
    test_camera_array_from_image_sizes()

    logger.info("test_capture_volume_save_load_roundtrip")
    test_capture_volume_save_load_roundtrip(debug_dir / "cv_roundtrip")

    logger.info("test_extract_image_points_missing_files")
    test_extract_image_points_missing_files()

    logger.info("test_extract_image_points_progress_callback")
    test_extract_image_points_progress_callback()

    logger.info("test_extract_image_points_includes_frame_time")
    test_extract_image_points_includes_frame_time()

    logger.info("test_calibrate_intrinsics_wrapper")
    test_calibrate_intrinsics_wrapper()

    logger.info("test_calibrate_intrinsics_missing_objloc")
    test_calibrate_intrinsics_missing_objloc()

    logger.info("All API tests passed.")
