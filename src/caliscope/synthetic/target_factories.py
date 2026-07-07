"""Calibration target factories bridging synthetic and production point layouts."""

from __future__ import annotations

import numpy as np

from caliscope.core.aruco_marker import ArucoMarker
from caliscope.synthetic.calibration_object import CalibrationObject

_FACE_NORMAL_Z = np.array([0.0, 0.0, 1.0])


def charuco_board(
    rows: int,
    cols: int,
    square_size: float,
    *,
    single_sided: bool = True,
) -> CalibrationObject:
    """Create a charuco board matching cv2.aruco.CharucoBoard.getChessboardCorners().

    Args:
        rows: Number of squares vertically
        cols: Number of squares horizontally
        square_size: Side length of each square (meters)
        single_sided: If True, set face_normal=(0,0,1) for visibility culling
    """
    n_inner_rows = rows - 1
    n_inner_cols = cols - 1
    n_corners = n_inner_rows * n_inner_cols

    points = np.zeros((n_corners, 3), dtype=np.float64)
    keypoint_ids = np.zeros(n_corners, dtype=np.int64)

    for row in range(n_inner_rows):
        for col in range(n_inner_cols):
            idx = row * n_inner_cols + col
            points[idx, 0] = (col + 1) * square_size
            points[idx, 1] = (row + 1) * square_size
            keypoint_ids[idx] = idx

    return CalibrationObject(
        points=points,
        keypoint_ids=keypoint_ids,
        face_normal=_FACE_NORMAL_Z.copy() if single_sided else None,
    )


def double_sided_charuco_board(
    rows: int,
    cols: int,
    square_size: float,
) -> CalibrationObject:
    """Charuco board visible from both sides (face_normal=None)."""
    return charuco_board(rows, cols, square_size, single_sided=False)


def box_target(width: float, height: float, depth: float) -> CalibrationObject:
    """Non-planar target: 8 box corners plus 6 face centers, centered at origin.

    14 genuinely non-coplanar points with no face_normal, so every camera sees
    all of them. Exercises the bootstrap's non-planar (SQPNP) PnP path, which
    the planar grid and charuco factories never reach.

    Args:
        width: Box extent along X (meters)
        height: Box extent along Y (meters)
        depth: Box extent along Z (meters)
    """
    half_width, half_height, half_depth = width / 2, height / 2, depth / 2
    corners = [
        [sx * half_width, sy * half_height, sz * half_depth] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]
    face_centers = [
        [-half_width, 0.0, 0.0],
        [half_width, 0.0, 0.0],
        [0.0, -half_height, 0.0],
        [0.0, half_height, 0.0],
        [0.0, 0.0, -half_depth],
        [0.0, 0.0, half_depth],
    ]
    points = np.array(corners + face_centers, dtype=np.float64)
    return CalibrationObject.from_points(points)


def aruco_marker(
    size: float,
    *,
    single_sided: bool = True,
) -> CalibrationObject:
    """Single ArUco marker matching ArucoMarker.corners geometry.

    Args:
        size: Marker side length (meters)
        single_sided: If True, set face_normal=(0,0,1) for visibility culling
    """
    marker = ArucoMarker(marker_id=0, size_m=size)
    return CalibrationObject(
        points=np.asarray(marker.corners, dtype=np.float64),
        keypoint_ids=np.arange(4, dtype=np.int64),
        face_normal=_FACE_NORMAL_Z.copy() if single_sided else None,
    )
