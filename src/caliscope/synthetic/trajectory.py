"""Trajectory representing object motion through space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.se3_pose import SE3Pose


@dataclass(frozen=True)
class Trajectory:
    """Sequence of SE3 poses representing object motion through space.

    The `origin_frame` is the frame where object coordinates = world coordinates.
    This is used by align_to_object() to resolve gauge freedom after optimization.

    Attributes:
        poses: Tuple of SE3Pose, one per frame (immutable)
        origin_frame: Frame index where pose is identity (object frame = world frame)
    """

    poses: tuple[SE3Pose, ...]
    origin_frame: int

    def __post_init__(self) -> None:
        """Validate trajectory."""
        if len(self.poses) == 0:
            raise ValueError("Trajectory must have at least one frame")
        if not (0 <= self.origin_frame < len(self.poses)):
            raise ValueError(f"origin_frame {self.origin_frame} out of range [0, {len(self.poses)})")

    def __len__(self) -> int:
        """Number of frames in the trajectory."""
        return len(self.poses)

    def __getitem__(self, frame: int) -> SE3Pose:
        """Get pose at frame index.

        Args:
            frame: Frame index (0 to len-1)

        Raises:
            IndexError: If frame out of range (including negative indices)
        """
        if not (0 <= frame < len(self.poses)):
            raise IndexError(f"Frame {frame} out of range [0, {len(self.poses)})")
        return self.poses[frame]

    @property
    def last(self) -> SE3Pose:
        """Return the last frame's pose. Convenient alternative to traj[len(traj)-1]."""
        return self.poses[-1]

    def world_points_at_frame(
        self,
        obj: CalibrationObject,
        frame: int,
    ) -> NDArray[np.float64]:
        """Transform object points to world coordinates at given frame.

        Args:
            obj: CalibrationObject with local point coordinates
            frame: Frame index

        Returns:
            (N, 3) array of points in world coordinates
        """
        pose = self[frame]
        return pose.apply(obj.points)

    @classmethod
    def orbital(
        cls,
        n_frames: int,
        radius_mm: float,
        arc_extent_deg: float = 360.0,
        height_mm: float = 0.0,
        tumble_rate: float = 0.0,
        origin_frame: int = 0,
    ) -> Trajectory:
        """Object orbits around world origin with optional tumble.

        The object center follows a circular arc in the XY plane at Z=height_mm.
        Optionally, the object rotates (tumbles) around its local Z-axis as it orbits.

        Args:
            n_frames: Number of frames in trajectory
            radius_mm: Distance from world origin to object center
            arc_extent_deg: Angular extent of orbit (360 = full circle, 180 = half)
            height_mm: Height of orbital plane above XY (Z coordinate)
            tumble_rate: Full rotations of object per orbit (0 = no tumble)
            origin_frame: Frame where object pose is identity

        Returns:
            Trajectory with n_frames poses

        Raises:
            ValueError: If n_frames < 1 or radius_mm <= 0
        """
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        if radius_mm <= 0:
            raise ValueError(f"radius_mm must be positive, got {radius_mm}")

        poses = []
        arc_rad = np.radians(arc_extent_deg)

        for i in range(n_frames):
            # For full circles (360°), don't include endpoint (it duplicates start)
            # For partial arcs, include endpoint for proper coverage
            if arc_extent_deg >= 360.0:
                # Periodic: 8 frames → angles 0°, 45°, 90°, ... 315° (not 360°)
                t = i / n_frames
            else:
                # Endpoint-inclusive: 5 frames over 180° → 0°, 45°, 90°, 135°, 180°
                t = i / max(n_frames - 1, 1)

            # Orbital position (object center in world coords)
            angle = arc_rad * t
            x = radius_mm * np.cos(angle)
            y = radius_mm * np.sin(angle)
            z = height_mm

            # Tumble rotation (around object's local Z-axis)
            tumble_angle = 2 * np.pi * tumble_rate * t

            # Build pose: translate to orbital position, then tumble
            # Start with identity, apply tumble, then translate
            tumble_pose = SE3Pose.from_axis_angle(
                axis=np.array([0, 0, 1], dtype=np.float64),
                angle_rad=tumble_angle,
                translation=np.array([x, y, z], dtype=np.float64),
            )
            poses.append(tumble_pose)

        # Adjust so origin_frame has identity pose
        origin_pose = poses[origin_frame]
        origin_inv = origin_pose.inverse()

        adjusted_poses = tuple(origin_inv.compose(p) for p in poses)

        return cls(poses=adjusted_poses, origin_frame=origin_frame)

    @classmethod
    def linear(
        cls,
        n_frames: int,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        tumble_rate: float = 0.0,
        origin_frame: int = 0,
    ) -> Trajectory:
        """Object moves in straight line from start to end.

        Args:
            n_frames: Number of frames in trajectory
            start: (3,) starting position in world coordinates
            end: (3,) ending position in world coordinates
            tumble_rate: Full rotations around Z-axis over trajectory
            origin_frame: Frame where object pose is identity

        Returns:
            Trajectory with n_frames poses

        Raises:
            ValueError: If n_frames < 1
        """
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")

        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)

        poses = []
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)

            # Linear interpolation
            position = (1 - t) * start + t * end

            # Tumble rotation
            tumble_angle = 2 * np.pi * tumble_rate * t

            pose = SE3Pose.from_axis_angle(
                axis=np.array([0, 0, 1], dtype=np.float64),
                angle_rad=tumble_angle,
                translation=position,
            )
            poses.append(pose)

        # Adjust so origin_frame has identity pose
        origin_pose = poses[origin_frame]
        origin_inv = origin_pose.inverse()

        adjusted_poses = tuple(origin_inv.compose(p) for p in poses)

        return cls(poses=adjusted_poses, origin_frame=origin_frame)

    @classmethod
    def stationary(cls, n_frames: int, pose: SE3Pose | None = None) -> Trajectory:
        """Object remains stationary at given pose for all frames.

        Useful for testing static scenes or single-frame calibration.

        Args:
            n_frames: Number of frames
            pose: Pose for all frames (default: identity)

        Returns:
            Trajectory with identical pose at all frames
        """
        if pose is None:
            pose = SE3Pose.identity()

        return cls(poses=tuple([pose] * n_frames), origin_frame=0)
