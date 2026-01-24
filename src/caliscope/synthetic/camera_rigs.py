"""Factory functions for common camera rig arrangements."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.synthetic.se3_pose import SE3Pose


def _default_intrinsics() -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]]:
    """Default camera intrinsics: 1920x1080, f=800px, zero distortion."""
    size = (1920, 1080)
    w, h = size
    f = 800.0

    matrix = np.array(
        [
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    distortions = np.zeros(5, dtype=np.float64)

    return matrix, distortions, size


def ring_rig(
    n_cameras: int,
    radius_mm: float,
    height_mm: float = 0.0,
    facing: Literal["inward", "outward"] = "inward",
    intrinsics: tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]] | None = None,
) -> CameraArray:
    """Create cameras evenly spaced in a ring, all at same height.

    Cameras are placed at equally-spaced angles around a circle of given radius,
    all at the specified height (Z coordinate). Each camera faces toward or away
    from the world origin.

    Args:
        n_cameras: Number of cameras in the ring (>= 2)
        radius_mm: Distance from world origin to each camera
        height_mm: Height of all cameras above XY plane
        facing: Direction cameras face
            - "inward": All cameras look toward world origin (typical setup)
            - "outward": All cameras look away from origin
        intrinsics: Optional (matrix, distortions, size) tuple. Uses defaults if None.

    Returns:
        CameraArray with n_cameras cameras, ports 0 to n_cameras-1

    Raises:
        ValueError: If n_cameras < 2 or radius_mm <= 0
    """
    if n_cameras < 2:
        raise ValueError(f"Need at least 2 cameras for a ring, got {n_cameras}")
    if radius_mm <= 0:
        raise ValueError(f"radius_mm must be positive, got {radius_mm}")

    if intrinsics is None:
        matrix, distortions, size = _default_intrinsics()
    else:
        matrix, distortions, size = intrinsics

    cameras: dict[int, CameraData] = {}

    for i in range(n_cameras):
        angle = 2 * np.pi * i / n_cameras

        # Camera position in world coordinates
        cam_x = radius_mm * np.cos(angle)
        cam_y = radius_mm * np.sin(angle)
        cam_z = height_mm
        position = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

        # Target for camera to look at
        if facing == "inward":
            target = np.array([0, 0, cam_z], dtype=np.float64)  # Look at origin (same height)
        else:
            # Look away from origin
            target = 2 * position - np.array([0, 0, cam_z], dtype=np.float64)

        # Create pose using look_at
        pose = SE3Pose.look_at(position, target)

        # Convert to OpenCV camera convention (rotation and translation)
        # OpenCV: point_camera = R @ point_world + t
        # Our SE3Pose.rotation transforms camera-to-world, we need world-to-camera
        rotation = pose.rotation  # Already camera-from-world (rows are camera axes)
        translation = -rotation @ position  # Camera translation

        cameras[i] = CameraData(
            port=i,
            size=size,
            matrix=matrix.copy(),
            distortions=distortions.copy(),
            rotation=rotation,
            translation=translation,
        )

    return CameraArray(cameras=cameras)


def linear_rig(
    n_cameras: int,
    spacing_mm: float,
    curvature: float = 0.0,
    intrinsics: tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]] | None = None,
) -> CameraArray:
    """Create cameras in a line, optionally curved into an arc.

    Cameras are placed along the X-axis, all looking toward -Y direction.
    With curvature > 0, the line curves toward the origin (into an arc).

    Args:
        n_cameras: Number of cameras (>= 2)
        spacing_mm: Distance between adjacent cameras
        curvature: Curve factor (0 = straight line, higher = more curved)
            - 0.0: Straight line along X-axis
            - 1.0: Moderate curve toward origin
            - Higher values: Tighter curve
        intrinsics: Optional (matrix, distortions, size) tuple

    Returns:
        CameraArray with n_cameras cameras

    Raises:
        ValueError: If n_cameras < 2 or spacing_mm <= 0
    """
    if n_cameras < 2:
        raise ValueError(f"Need at least 2 cameras, got {n_cameras}")
    if spacing_mm <= 0:
        raise ValueError(f"spacing_mm must be positive, got {spacing_mm}")

    if intrinsics is None:
        matrix, distortions, size = _default_intrinsics()
    else:
        matrix, distortions, size = intrinsics

    cameras: dict[int, CameraData] = {}

    # Center the line at x=0
    total_width = (n_cameras - 1) * spacing_mm
    start_x = -total_width / 2

    for i in range(n_cameras):
        # Base position along X-axis
        x = start_x + i * spacing_mm

        # Apply curvature: push camera back along Y based on distance from center
        # Parabolic curve: y = -curvature * (x^2) / (total_width^2) * baseline_distance
        baseline_y = -2000  # 2m away by default
        if curvature > 0 and total_width > 0:
            normalized_x = x / (total_width / 2) if total_width > 0 else 0
            y = baseline_y - curvature * 500 * (normalized_x**2)
        else:
            y = baseline_y

        position = np.array([x, y, 0], dtype=np.float64)
        target = np.array([x, 0, 0], dtype=np.float64)  # Look toward origin's X-plane

        pose = SE3Pose.look_at(position, target)
        rotation = pose.rotation
        translation = -rotation @ position

        cameras[i] = CameraData(
            port=i,
            size=size,
            matrix=matrix.copy(),
            distortions=distortions.copy(),
            rotation=rotation,
            translation=translation,
        )

    return CameraArray(cameras=cameras)


def nested_rings_rig(
    inner_n: int,
    outer_n: int,
    inner_radius_mm: float,
    outer_radius_mm: float,
    inner_height_mm: float = 0.0,
    outer_height_mm: float = 500.0,
    intrinsics: tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int]] | None = None,
) -> CameraArray:
    """Create two concentric camera rings.

    Inner ring faces outward, outer ring faces inward. Useful for testing
    configurations with cameras that don't directly see each other.

    Args:
        inner_n: Number of cameras in inner ring
        outer_n: Number of cameras in outer ring
        inner_radius_mm: Radius of inner ring
        outer_radius_mm: Radius of outer ring (must be > inner)
        inner_height_mm: Height of inner ring cameras
        outer_height_mm: Height of outer ring cameras
        intrinsics: Optional (matrix, distortions, size) tuple

    Returns:
        CameraArray with inner_n + outer_n cameras
        Ports 0 to inner_n-1 are inner ring, inner_n to inner_n+outer_n-1 are outer

    Raises:
        ValueError: If radii invalid or camera counts < 2
    """
    if inner_n < 2 or outer_n < 2:
        raise ValueError(f"Need at least 2 cameras per ring, got {inner_n}, {outer_n}")
    if inner_radius_mm >= outer_radius_mm:
        raise ValueError(f"Inner radius must be < outer radius: {inner_radius_mm} >= {outer_radius_mm}")

    if intrinsics is None:
        matrix, distortions, size = _default_intrinsics()
    else:
        matrix, distortions, size = intrinsics

    cameras: dict[int, CameraData] = {}

    # Inner ring (facing outward)
    for i in range(inner_n):
        angle = 2 * np.pi * i / inner_n
        position = np.array(
            [
                inner_radius_mm * np.cos(angle),
                inner_radius_mm * np.sin(angle),
                inner_height_mm,
            ],
            dtype=np.float64,
        )

        # Look outward (away from origin)
        target = 2 * position
        target[2] = inner_height_mm

        pose = SE3Pose.look_at(position, target)
        rotation = pose.rotation
        translation = -rotation @ position

        cameras[i] = CameraData(
            port=i,
            size=size,
            matrix=matrix.copy(),
            distortions=distortions.copy(),
            rotation=rotation,
            translation=translation,
        )

    # Outer ring (facing inward)
    for i in range(outer_n):
        angle = 2 * np.pi * i / outer_n
        position = np.array(
            [
                outer_radius_mm * np.cos(angle),
                outer_radius_mm * np.sin(angle),
                outer_height_mm,
            ],
            dtype=np.float64,
        )

        # Look inward (toward origin at same height)
        target = np.array([0, 0, outer_height_mm], dtype=np.float64)

        pose = SE3Pose.look_at(position, target)
        rotation = pose.rotation
        translation = -rotation @ position

        port = inner_n + i
        cameras[port] = CameraData(
            port=port,
            size=size,
            matrix=matrix.copy(),
            distortions=distortions.copy(),
            rotation=rotation,
            translation=translation,
        )

    return CameraArray(cameras=cameras)


def strip_extrinsics(camera_array: CameraArray) -> CameraArray:
    """Return a copy of camera_array with extrinsics removed.

    Used to create the "intrinsics-only" input for the calibration pipeline.

    Args:
        camera_array: Source camera array (not modified)

    Returns:
        New CameraArray with rotation=None, translation=None for all cameras
    """
    cameras: dict[int, CameraData] = {}

    for port, cam in camera_array.cameras.items():
        cameras[port] = CameraData(
            port=cam.port,
            size=cam.size,
            rotation_count=cam.rotation_count,
            error=cam.error,
            matrix=cam.matrix.copy() if cam.matrix is not None else None,
            distortions=cam.distortions.copy() if cam.distortions is not None else None,
            exposure=cam.exposure,
            grid_count=cam.grid_count,
            ignore=cam.ignore,
            fisheye=cam.fisheye,
            translation=None,
            rotation=None,
        )

    return CameraArray(cameras=cameras)
