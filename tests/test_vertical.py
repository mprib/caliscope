"""Tests for the vertical estimator: preprocessing contract and a gated run.

The preprocessing tests pin the GeoCalib input geometry (short side 320, edges
divisible by 32, resize scales, dtype/range) without touching the network. The
integration test runs the real field net and is skipped unless the weights are
already present in MODELS_DIR -- it never downloads the 118 MB model.
"""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.estimators.vertical import (
    GEOCALIB_FIELDS_SPEC,
    VerticalEstimate,
    _net_size,
    estimate_vertical,
    preprocess_frame,
    sample_frame_indices,
)
from caliscope.recording.video_utils import read_video_properties

TESTS_ROOT = __import__("pathlib").Path(__file__).parent
EXTRINSIC_VIDEOS = {
    0: TESTS_ROOT / "sessions" / "h264_extrinsic" / "cam_0.mp4",
    1: TESTS_ROOT / "sessions" / "h264_extrinsic" / "cam_1.mp4",
}


def _synthetic_frame(height: int, width: int) -> np.ndarray:
    """A BGR frame with spatial structure so downsampling has something to average."""
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    ys = np.linspace(0, 255, height, dtype=np.uint8)
    blue = np.broadcast_to(xs, (height, width))
    green = np.broadcast_to(ys[:, None], (height, width))
    red = ((blue.astype(int) + green.astype(int)) // 2).astype(np.uint8)
    return np.stack([blue, green, red], axis=-1).copy()


@pytest.mark.parametrize(
    "height, width, expected_net",
    [
        (1080, 1920, (320, 568)),  # landscape: short side is height
        (1920, 1080, (568, 320)),  # portrait: short side is width
        (1000, 1000, (320, 320)),  # square
    ],
)
def test_net_size_short_side(height: int, width: int, expected_net: tuple[int, int]) -> None:
    assert _net_size(height, width) == expected_net
    net_h, net_w = _net_size(height, width)
    assert min(net_h, net_w) == 320


def test_preprocess_shape_divisible_by_32() -> None:
    frame = _synthetic_frame(1080, 1920)
    image, _, _ = preprocess_frame(frame)

    assert image.shape[0] == 1
    assert image.shape[1] == 3
    net_h, net_w = image.shape[2], image.shape[3]
    assert net_h % 32 == 0
    assert net_w % 32 == 0
    # Short side (320) is already a multiple of 32, so it survives the crop.
    assert min(net_h, net_w) == 320


def test_preprocess_dtype_and_range() -> None:
    frame = _synthetic_frame(720, 1280)
    image, _, _ = preprocess_frame(frame)

    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_preprocess_scales_are_net_over_original() -> None:
    frame = _synthetic_frame(1080, 1920)
    _, scale_x, scale_y = preprocess_frame(frame)

    # net_w = int(320 * 1920 / 1080) = 568; net_h = 320. Scales are pre-crop.
    assert np.isclose(scale_x, 568 / 1920)
    assert np.isclose(scale_y, 320 / 1080)


def test_preprocess_rgb_channel_order() -> None:
    # A pure-blue BGR frame (B=255) must land in the last RGB channel.
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    frame[..., 0] = 255  # blue in BGR
    image, _, _ = preprocess_frame(frame)

    assert image[0, 2].mean() > 0.9  # blue -> RGB channel 2
    assert image[0, 0].mean() < 0.1  # red channel empty


def test_sample_frame_indices_spans_clip() -> None:
    indices = sample_frame_indices(100, 12)
    assert indices[0] == 0
    assert indices[-1] == 99
    assert list(indices) == sorted(set(indices))
    assert len(indices) == 12

    # Fewer frames than samples returns every frame.
    assert sample_frame_indices(5, 12) == (0, 1, 2, 3, 4)


def _camera_array_for_videos() -> CameraArray:
    """Build a CameraArray with a synthesized focal prior for the test videos."""
    cameras = {}
    for cam_id, path in EXTRINSIC_VIDEOS.items():
        props = read_video_properties(path)
        w, h = props["width"], props["height"]
        matrix = np.array([[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]])
        cameras[cam_id] = CameraData(cam_id=cam_id, size=(w, h), matrix=matrix)
    return CameraArray(cameras=cameras)


@pytest.mark.skipif(
    not GEOCALIB_FIELDS_SPEC.model_path.exists(),
    reason="GeoCalib field net not present in MODELS_DIR (not downloaded in tests)",
)
@pytest.mark.skipif(
    not all(path.exists() for path in EXTRINSIC_VIDEOS.values()),
    reason="extrinsic test videos not available",
)
def test_estimate_vertical_runs_on_real_videos() -> None:
    cameras = _camera_array_for_videos()
    estimate = estimate_vertical(EXTRINSIC_VIDEOS, cameras, frames_per_camera=2)

    assert isinstance(estimate, VerticalEstimate)
    assert set(estimate.up_per_cam) == set(EXTRINSIC_VIDEOS)
    for cam_id in EXTRINSIC_VIDEOS:
        up = estimate.up_per_cam[cam_id]
        assert up.shape == (3,)
        assert np.isclose(np.linalg.norm(up), 1.0)
        # Frame-to-frame spread should be small and non-negative on real footage.
        assert 0.0 <= estimate.spread_per_cam[cam_id] < 5.0


if __name__ == "__main__":
    frame = _synthetic_frame(1080, 1920)
    image, sx, sy = preprocess_frame(frame)
    print(f"preprocessed {frame.shape} -> {image.shape}, scales ({sx:.4f}, {sy:.4f})")
