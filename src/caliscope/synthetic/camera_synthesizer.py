"""Fluent builder for synthetic camera arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.synthetic.se3_pose import SE3Pose


@dataclass
class _CameraSpec:
    """Internal: specification for a single camera before building."""

    port: int
    position: NDArray[np.float64]
    target: NDArray[np.float64]
    roll_deg: float = 0.0
    pitch_deg: float = 0.0


class CameraSynthesizer:
    """Fluent builder for synthetic camera arrays.

    Enables composable camera rig construction with orientation variation
    and port gaps (simulating disconnected cameras).

    Example:
        array = (
            CameraSynthesizer()
            .add_ring(n=4, radius_mm=2000, height_mm=0)
            .add_ring(n=4, radius_mm=2000, height_mm=500, angular_offset_deg=45)
            .drop_ports(1, 5)
            .build()
        )
        # Result: CameraArray with ports [0, 2, 3, 4, 6, 7]
    """

    def __init__(self) -> None:
        self._specs: list[_CameraSpec] = []
        self._next_port: int = 0
        self._dropped: set[int] = set()

    def add_ring(
        self,
        n: int,
        radius_mm: float,
        height_mm: float = 0.0,
        facing: Literal["inward", "outward"] = "inward",
        angular_offset_deg: float = 0.0,
        roll_variation_deg: float = 0.0,
        pitch_variation_deg: float = 0.0,
        random_seed: int = 42,
    ) -> CameraSynthesizer:
        """Add a ring of cameras at specified height.

        Args:
            n: Number of cameras in this ring (>= 1)
            radius_mm: Distance from world origin to each camera
            height_mm: Z-coordinate of all cameras in this ring
            facing: Direction cameras point ("inward" toward origin, "outward" away)
            angular_offset_deg: Rotate the entire ring by this angle (useful for
                staggering multiple rings)
            roll_variation_deg: Random roll within +/- this range (degrees)
            pitch_variation_deg: Random pitch within +/- this range (degrees)
            random_seed: Seed for reproducible random orientation variation

        Returns:
            self, for method chaining
        """
        rng = np.random.default_rng(random_seed)
        offset_rad = np.radians(angular_offset_deg)

        for i in range(n):
            angle = 2 * np.pi * i / n + offset_rad
            position = np.array(
                [
                    radius_mm * np.cos(angle),
                    radius_mm * np.sin(angle),
                    height_mm,
                ],
                dtype=np.float64,
            )

            if facing == "inward":
                target = np.array([0, 0, height_mm], dtype=np.float64)
            else:
                # Look away from origin at same height
                target = 2 * position - np.array([0, 0, height_mm], dtype=np.float64)

            roll = rng.uniform(-roll_variation_deg, roll_variation_deg) if roll_variation_deg else 0.0
            pitch = rng.uniform(-pitch_variation_deg, pitch_variation_deg) if pitch_variation_deg else 0.0

            self._specs.append(
                _CameraSpec(
                    port=self._next_port,
                    position=position,
                    target=target,
                    roll_deg=roll,
                    pitch_deg=pitch,
                )
            )
            self._next_port += 1

        return self

    def add_line(
        self,
        n: int,
        spacing_mm: float,
        distance_mm: float = 2000.0,
        height_mm: float = 0.0,
        curvature: float = 0.0,
        roll_variation_deg: float = 0.0,
        pitch_variation_deg: float = 0.0,
        random_seed: int = 42,
    ) -> CameraSynthesizer:
        """Add a line of cameras.

        Cameras are placed along the X-axis, all looking toward the Y=0 plane.
        With curvature > 0, the line curves toward the origin (into an arc).

        Args:
            n: Number of cameras in this line (>= 1)
            spacing_mm: Distance between adjacent cameras
            distance_mm: Distance from origin along -Y axis
            height_mm: Z-coordinate of all cameras in this line
            curvature: Curve factor (0 = straight, higher = more curved toward origin)
            roll_variation_deg: Random roll within +/- this range (degrees)
            pitch_variation_deg: Random pitch within +/- this range (degrees)
            random_seed: Seed for reproducible random orientation variation

        Returns:
            self, for method chaining
        """
        rng = np.random.default_rng(random_seed)
        total_width = (n - 1) * spacing_mm
        start_x = -total_width / 2

        for i in range(n):
            x = start_x + i * spacing_mm
            y = -distance_mm

            # Apply parabolic curvature
            if curvature > 0 and total_width > 0:
                normalized_x = x / (total_width / 2)
                y -= curvature * 500 * (normalized_x**2)

            position = np.array([x, y, height_mm], dtype=np.float64)
            target = np.array([x, 0, height_mm], dtype=np.float64)

            roll = rng.uniform(-roll_variation_deg, roll_variation_deg) if roll_variation_deg else 0.0
            pitch = rng.uniform(-pitch_variation_deg, pitch_variation_deg) if pitch_variation_deg else 0.0

            self._specs.append(
                _CameraSpec(
                    port=self._next_port,
                    position=position,
                    target=target,
                    roll_deg=roll,
                    pitch_deg=pitch,
                )
            )
            self._next_port += 1

        return self

    def drop_ports(self, *ports: int) -> CameraSynthesizer:
        """Exclude ports from final array (creates gaps in numbering).

        Useful for simulating disconnected cameras or testing sparse configurations.

        Args:
            *ports: Port numbers to exclude from the final CameraArray

        Returns:
            self, for method chaining
        """
        self._dropped.update(ports)
        return self

    def build(self) -> CameraArray:
        """Build the CameraArray from accumulated specs.

        Applies roll/pitch variations and excludes dropped ports.

        Returns:
            CameraArray with cameras at non-dropped ports

        Raises:
            ValueError: If fewer than 2 cameras remain after dropping ports
        """
        cameras: dict[int, CameraData] = {}

        for spec in self._specs:
            if spec.port in self._dropped:
                continue

            pose = SE3Pose.look_at(spec.position, spec.target)

            if spec.pitch_deg != 0:
                pose = pose.with_pitch(np.radians(spec.pitch_deg))
            if spec.roll_deg != 0:
                pose = pose.with_roll(np.radians(spec.roll_deg))

            rotation = pose.rotation
            # OpenCV convention: t = -R @ position
            translation = -rotation @ spec.position

            cameras[spec.port] = CameraData(
                port=spec.port,
                size=(1920, 1080),
                matrix=_default_matrix(),
                distortions=np.zeros(5, dtype=np.float64),
                rotation=rotation,
                translation=translation,
            )

        if len(cameras) < 2:
            raise ValueError(
                f"Need at least 2 cameras for calibration, got {len(cameras)} (dropped ports: {sorted(self._dropped)})"
            )

        return CameraArray(cameras=cameras)


def _default_matrix() -> NDArray[np.float64]:
    """Default camera matrix: 1920x1080, f=800px, principal point at center."""
    return np.array(
        [
            [800, 0, 960],
            [0, 800, 540],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def strip_extrinsics(camera_array: CameraArray) -> CameraArray:
    """Return a copy of camera_array with extrinsics removed.

    Used to create the "intrinsics-only" input for the calibration pipeline,
    simulating uncalibrated cameras that have known intrinsics but need
    extrinsic calibration.

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
