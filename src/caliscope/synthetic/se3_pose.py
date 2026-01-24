"""SE(3) pose representation for rigid body transformations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SE3Pose:
    """Rigid body pose in 3D space (rotation + translation).

    Represents "where is this object, and which way is it facing?"
    Immutable - all operations return new instances.

    Attributes:
        rotation: (3, 3) rotation matrix (orthonormal, det=+1)
        translation: (3,) translation vector in world units (mm)
    """

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate rotation matrix properties."""
        if self.rotation.shape != (3, 3):
            raise ValueError(f"Rotation must be 3x3, got {self.rotation.shape}")
        if self.translation.shape != (3,):
            raise ValueError(f"Translation must be 3-vector, got {self.translation.shape}")

        det = np.linalg.det(self.rotation)
        if not np.isclose(det, 1.0, atol=1e-6):
            raise ValueError(f"Rotation must be proper (det=+1), got det={det:.6f}")

        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("Rotation matrix must be orthogonal (R @ R.T = I)")

    @classmethod
    def identity(cls) -> SE3Pose:
        """No rotation, no translation (object at world origin, aligned with world axes)."""
        return cls(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )

    @classmethod
    def from_matrix(cls, matrix: NDArray[np.float64]) -> SE3Pose:
        """Extract from 4x4 homogeneous transformation matrix.

        Args:
            matrix: 4x4 array where top-left 3x3 is rotation, top-right 3x1 is translation

        Raises:
            ValueError: If matrix shape is not 4x4 or bottom row is not [0,0,0,1]
        """
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got {matrix.shape}")
        if not np.allclose(matrix[3, :], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("Bottom row must be [0, 0, 0, 1]")

        return cls(
            rotation=matrix[:3, :3].copy(),
            translation=matrix[:3, 3].copy(),
        )

    @classmethod
    def from_axis_angle(
        cls,
        axis: NDArray[np.float64],
        angle_rad: float,
        translation: NDArray[np.float64],
    ) -> SE3Pose:
        """Construct from axis-angle rotation representation.

        Args:
            axis: (3,) unit vector defining rotation axis
            angle_rad: Rotation angle in radians (right-hand rule)
            translation: (3,) translation vector

        Raises:
            ValueError: If axis is not unit length
        """
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if not np.isclose(axis_norm, 1.0, atol=1e-6):
            raise ValueError(f"Axis must be unit vector, got norm={axis_norm}")

        # Rodrigues' rotation formula
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ],
            dtype=np.float64,
        )

        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation = np.eye(3) + s * K + (1 - c) * (K @ K)

        return cls(rotation=rotation, translation=np.asarray(translation, dtype=np.float64))

    @classmethod
    def look_at(
        cls,
        position: NDArray[np.float64],
        target: NDArray[np.float64],
        up: NDArray[np.float64] | None = None,
    ) -> SE3Pose:
        """Create pose at position, looking toward target.

        Useful for placing cameras that face a central point.

        Args:
            position: (3,) world position of the object
            target: (3,) point the object should face
            up: (3,) world up vector for orientation (default: Z-up [0, 0, 1])

        Returns:
            SE3Pose with Z-axis pointing from position toward target

        Raises:
            ValueError: If position equals target (undefined direction)
        """
        position = np.asarray(position, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if up is None:
            up = np.array([0, 0, 1], dtype=np.float64)
        else:
            up = np.asarray(up, dtype=np.float64)

        # Forward direction (object Z-axis in world)
        forward = target - position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-9:
            raise ValueError("Position must not equal target (undefined direction)")
        forward = forward / forward_norm

        # Right direction (object X-axis in world)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-9:
            raise ValueError("Forward and up vectors must not be parallel")
        right = right / right_norm

        # Recompute up to ensure orthogonality (object Y-axis, but inverted for camera convention)
        down = np.cross(forward, right)
        down = down / np.linalg.norm(down)

        # Rotation matrix: rows are camera axes in world coordinates
        # Camera convention: X=right, Y=down, Z=forward (OpenCV)
        rotation = np.vstack([right, down, forward])

        return cls(rotation=rotation, translation=position)

    @cached_property
    def matrix(self) -> NDArray[np.float64]:
        """Return 4x4 homogeneous transformation matrix.

        Format: [[R, t], [0, 0, 0, 1]]
        """
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def compose(self, other: SE3Pose) -> SE3Pose:
        """Chain transformations: self then other.

        If self transforms A->B and other transforms B->C, result transforms A->C.

        Args:
            other: Transform to apply after self

        Returns:
            New SE3Pose representing composed transformation
        """
        composed = other.matrix @ self.matrix
        return SE3Pose.from_matrix(composed)

    def inverse(self) -> SE3Pose:
        """Return the inverse transformation.

        If self transforms A->B, inverse transforms B->A.
        """
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return SE3Pose(rotation=inv_rotation, translation=inv_translation)

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform points from local frame to this pose's frame.

        Args:
            points: (N, 3) array of points in local coordinates

        Returns:
            (N, 3) array of points in world coordinates

        Raises:
            ValueError: If points shape is not (N, 3)
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3), got shape {points.shape}")

        return (self.rotation @ points.T).T + self.translation
