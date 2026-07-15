"""Tests for the MoGe runner and focal-based CameraArray construction.

The full ``run_moge`` integration test is skipped unless the (>1 GB) model is
already present in MODELS_DIR — it is never downloaded here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints
from caliscope.estimators.moge import MOGE_MODEL_SPEC, run_moge

SESSION = Path(__file__).parent / "sessions" / "4_cam_recording" / "calibration" / "extrinsic"


def _videos() -> dict[int, Path]:
    return {cam_id: SESSION / f"cam_{cam_id}.mp4" for cam_id in (0, 1, 2, 3)}


def test_from_focal_estimates_builds_pinhole_intrinsics():
    videos = _videos()
    focal_per_cam = {0: 512.0, 2: 640.0}

    cameras = CameraArray.from_focal_estimates(focal_per_cam, videos)

    assert set(cameras.cameras.keys()) == {0, 2}
    for cam_id, focal in focal_per_cam.items():
        cam = cameras[cam_id]
        width, height = cam.size
        assert cam.matrix is not None
        assert cam.matrix[0, 0] == pytest.approx(focal)
        assert cam.matrix[1, 1] == pytest.approx(focal)
        assert cam.matrix[0, 2] == pytest.approx(width / 2.0)
        assert cam.matrix[1, 2] == pytest.approx(height / 2.0)
        assert cam.distortions is not None and np.allclose(cam.distortions, 0.0)
        # intrinsics only — no pose
        assert cam.rotation is None and cam.translation is None


def test_from_focal_estimates_missing_video_raises():
    with pytest.raises(ValueError, match="No video provided for cam_id 5"):
        CameraArray.from_focal_estimates({5: 500.0}, _videos())


def _pinhole_point_map(height: int, width: int, focal_px: float) -> np.ndarray:
    """A valid pinhole point map so recover_focal_shift returns a sane focal + positive depth."""
    j = np.arange(width, dtype=np.float64)
    i = np.arange(height, dtype=np.float64)
    jj, ii = np.meshgrid(j, i, indexing="xy")
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    depth = 3.0 + 0.002 * ii + 0.001 * jj
    x = (jj - cx) * depth / focal_px
    y = (ii - cy) * depth / focal_px
    return np.stack([x, y, depth], axis=-1).astype(np.float32)


def test_run_moge_decodes_by_frame_index_but_keys_by_sync_index(monkeypatch):
    """When frame_index diverges from sync_index, decode by frame_index, emit by sync_index."""
    import caliscope.estimators.moge as moge

    requested_frames: dict[int, list[int]] = {}

    def fake_decode(video_path, cam_id, frame_indices):
        requested_frames[cam_id] = list(frame_indices)
        return {fi: np.zeros((32, 32, 3), dtype=np.uint8) for fi in frame_indices}

    def fake_infer(session, rgb):
        h, w = rgb.shape[:2]
        return _pinhole_point_map(h, w, 500.0), np.ones((h, w), dtype=bool), 1.0

    monkeypatch.setattr(moge, "_build_session", lambda: object())
    monkeypatch.setattr(moge, "_decode_frames", fake_decode)
    monkeypatch.setattr(moge, "_infer_frame", fake_infer)

    # frame_index is offset from sync_index by +2 (staggered start / dropped frames).
    rows = []
    for sync_index in (0, 5, 10, 15):
        for keypoint_id in range(2):
            rows.append(
                {
                    "sync_index": sync_index,
                    "cam_id": 0,
                    "object_id": 0,
                    "keypoint_id": keypoint_id,
                    "img_loc_x": 12.0 + keypoint_id,
                    "img_loc_y": 12.0,
                    "frame_index": sync_index + 2,
                }
            )
    points = ImagePoints(pd.DataFrame(rows))

    result = run_moge({0: SESSION / "cam_0.mp4"}, points, frames_per_camera=4)

    # Decoding requested frame_index values, not sync_index values.
    assert sorted(requested_frames[0]) == [2, 7, 12, 17]

    # Observations carry sync_index, never frame_index.
    emitted_syncs = {obs.sync_index for obs in result.depth_observations}
    assert emitted_syncs == {0, 5, 10, 15}
    assert emitted_syncs.isdisjoint({2, 7, 12, 17})
    assert 0 in result.focal_per_cam


def test_run_moge_without_frame_index_uses_sync_as_frame(monkeypatch):
    """Absent a frame_index column, sync_index is the frame index (single-video path)."""
    import caliscope.estimators.moge as moge

    requested_frames: dict[int, list[int]] = {}

    def fake_decode(video_path, cam_id, frame_indices):
        requested_frames[cam_id] = list(frame_indices)
        return {fi: np.zeros((32, 32, 3), dtype=np.uint8) for fi in frame_indices}

    def fake_infer(session, rgb):
        h, w = rgb.shape[:2]
        return _pinhole_point_map(h, w, 500.0), np.ones((h, w), dtype=bool), 1.0

    monkeypatch.setattr(moge, "_build_session", lambda: object())
    monkeypatch.setattr(moge, "_decode_frames", fake_decode)
    monkeypatch.setattr(moge, "_infer_frame", fake_infer)

    rows = [
        {
            "sync_index": sync_index,
            "cam_id": 0,
            "object_id": 0,
            "keypoint_id": 0,
            "img_loc_x": 12.0,
            "img_loc_y": 12.0,
        }
        for sync_index in (3, 8, 11)
    ]
    points = ImagePoints(pd.DataFrame(rows))

    result = run_moge({0: SESSION / "cam_0.mp4"}, points, frames_per_camera=4)

    assert sorted(requested_frames[0]) == [3, 8, 11]
    assert {obs.sync_index for obs in result.depth_observations} == {3, 8, 11}


@pytest.mark.skipif(
    not MOGE_MODEL_SPEC.model_path.exists(),
    reason="MoGe model not present in MODELS_DIR (>1 GB; not downloaded in tests)",
)
def test_run_moge_produces_focals_and_depths():
    videos = _videos()

    # A handful of fake keypoint detections near frame center for two cameras.
    rows = []
    for cam_id in (0, 1):
        for sync_index in range(0, 20, 5):
            for keypoint_id in range(3):
                rows.append(
                    {
                        "sync_index": sync_index,
                        "cam_id": cam_id,
                        "object_id": 0,
                        "keypoint_id": keypoint_id,
                        "img_loc_x": 300.0 + 20 * keypoint_id,
                        "img_loc_y": 240.0 + 10 * keypoint_id,
                    }
                )
    points = ImagePoints(pd.DataFrame(rows))

    result = run_moge(videos, points, frames_per_camera=2)

    assert set(result.focal_per_cam.keys()) <= {0, 1}
    assert all(f > 0 for f in result.focal_per_cam.values())
    for obs in result.depth_observations:
        assert obs.cam_id in (0, 1)
        assert obs.depth_m > 0
