"""Validate the vendored MoGe focal/shift recovery against a synthetic pinhole.

Builds a perfect pinhole point map from a known focal length (project a grid of
3D points through a pinhole; the point map is the camera-frame 3D coordinates per
pixel), then checks that ``recover_focal_shift_numpy`` + the runner's pixel
conversion recover the focal within tight tolerance. This exercises the whole
vendored chain (masked resize, view-plane UV, least squares) without the model.
"""

import numpy as np

from caliscope.estimators.moge import _focal_to_pixels
from caliscope.estimators.moge_utils import recover_focal_shift_numpy


def _synthetic_pinhole_point_map(width: int, height: int, focal_px: float) -> np.ndarray:
    """A (H, W, 3) point map for a pinhole camera with a tilted-plane depth.

    Principal point sits at ((W-1)/2, (H-1)/2), matching the center of MoGe's
    normalized view-plane UV grid, so the recovered focal maps back to ``focal_px``
    exactly. Depth varies across the frame (a tilted plane) to break the
    focal/shift degeneracy that a frontoparallel plane would leave.
    """
    j = np.arange(width, dtype=np.float64)
    i = np.arange(height, dtype=np.float64)
    jj, ii = np.meshgrid(j, i, indexing="xy")

    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    depth = 3.0 + 0.002 * ii + 0.001 * jj  # tilted plane, ~3 m

    x = (jj - cx) * depth / focal_px
    y = (ii - cy) * depth / focal_px
    return np.stack([x, y, depth], axis=-1).astype(np.float32)


def test_recover_focal_shift_matches_known_focal():
    width, height, focal_px = 640, 480, 500.0
    point_map = _synthetic_pinhole_point_map(width, height, focal_px)
    mask = np.ones((height, width), dtype=bool)

    focal, shift = recover_focal_shift_numpy(point_map, mask)
    recovered_px = _focal_to_pixels(float(focal), width, height)

    assert abs(recovered_px - focal_px) / focal_px < 0.005
    assert abs(float(shift)) < 0.01  # points already carry true depth; no shift needed


def test_recover_focal_shift_portrait_aspect():
    """Aspect-ratio handling in the pixel conversion holds for a portrait frame."""
    width, height, focal_px = 480, 640, 700.0
    point_map = _synthetic_pinhole_point_map(width, height, focal_px)
    mask = np.ones((height, width), dtype=bool)

    focal, _ = recover_focal_shift_numpy(point_map, mask)
    recovered_px = _focal_to_pixels(float(focal), width, height)

    assert abs(recovered_px - focal_px) / focal_px < 0.005


if __name__ == "__main__":
    test_recover_focal_shift_matches_known_focal()
    test_recover_focal_shift_portrait_aspect()
    print("moge_utils recovery validated")
