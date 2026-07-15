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
