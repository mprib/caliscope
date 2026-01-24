"""Tests for Trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.synthetic import CalibrationObject, SE3Pose, Trajectory


class TestConstruction:
    """Test Trajectory construction and validation."""

    def test_valid_trajectory_accepted(self):
        """Create trajectory from tuple of poses."""
        poses = (SE3Pose.identity(), SE3Pose.identity())
        traj = Trajectory(poses=poses, origin_frame=0)

        assert len(traj) == 2
        assert traj.origin_frame == 0

    def test_empty_trajectory_rejected(self):
        """Trajectory must have at least one frame."""
        with pytest.raises(ValueError, match="at least one frame"):
            Trajectory(poses=tuple(), origin_frame=0)


class TestIndexing:
    """Test Trajectory indexing and length."""

    def test_len_returns_number_of_frames(self):
        """len() returns number of poses."""
        poses = tuple(SE3Pose.identity() for _ in range(5))
        traj = Trajectory(poses=poses, origin_frame=0)

        assert len(traj) == 5

    def test_getitem_returns_pose(self):
        """Can index trajectory to get pose."""
        pose1 = SE3Pose.identity()
        pose2 = SE3Pose(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.array([10, 0, 0], dtype=np.float64),
        )
        traj = Trajectory(poses=(pose1, pose2), origin_frame=0)

        retrieved = traj[1]
        assert np.allclose(retrieved.translation, pose2.translation)


class TestWorldPointsAtFrame:
    """Test world_points_at_frame() transformation."""

    def test_identity_pose_leaves_points_unchanged(self):
        """Identity trajectory doesn't transform points."""
        obj = CalibrationObject.planar_grid(rows=2, cols=2, spacing_mm=10.0)
        traj = Trajectory.stationary(n_frames=1, pose=SE3Pose.identity())

        world_points = traj.world_points_at_frame(obj, frame=0)

        assert np.allclose(world_points, obj.points)


class TestOrbitalTrajectory:
    """Test Trajectory.orbital() factory method."""

    def test_creates_correct_number_of_frames(self):
        """Orbital trajectory has requested number of frames."""
        traj = Trajectory.orbital(n_frames=10, radius_mm=1000.0)

        assert len(traj) == 10

    def test_origin_frame_has_identity_pose(self):
        """Pose at origin_frame is identity."""
        traj = Trajectory.orbital(n_frames=5, radius_mm=1000.0, origin_frame=2)

        pose = traj[2]
        assert np.allclose(pose.rotation, np.eye(3), atol=1e-10)
        assert np.allclose(pose.translation, np.zeros(3), atol=1e-10)

    def test_invalid_radius_rejected(self):
        """Radius must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            Trajectory.orbital(n_frames=5, radius_mm=0.0)


class TestLinearTrajectory:
    """Test Trajectory.linear() factory method."""

    def test_creates_correct_number_of_frames(self):
        """Linear trajectory has requested number of frames."""
        start = np.array([0, 0, 0], dtype=np.float64)
        end = np.array([100, 100, 100], dtype=np.float64)

        traj = Trajectory.linear(n_frames=10, start=start, end=end)

        assert len(traj) == 10

    def test_origin_frame_has_identity_pose(self):
        """Pose at origin_frame is identity."""
        start = np.array([0, 0, 0], dtype=np.float64)
        end = np.array([100, 0, 0], dtype=np.float64)

        traj = Trajectory.linear(n_frames=5, start=start, end=end, origin_frame=2)

        pose = traj[2]
        assert np.allclose(pose.rotation, np.eye(3), atol=1e-10)
        assert np.allclose(pose.translation, np.zeros(3), atol=1e-10)


class TestStationaryTrajectory:
    """Test Trajectory.stationary() factory method."""

    def test_all_frames_have_same_pose(self):
        """Stationary trajectory has identical pose at all frames."""
        pose = SE3Pose(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.array([10, 20, 30], dtype=np.float64),
        )
        traj = Trajectory.stationary(n_frames=5, pose=pose)

        for i in range(len(traj)):
            assert np.allclose(traj[i].rotation, pose.rotation)
            assert np.allclose(traj[i].translation, pose.translation)

    def test_origin_frame_is_zero(self):
        """Stationary trajectory has origin_frame=0."""
        traj = Trajectory.stationary(n_frames=5)

        assert traj.origin_frame == 0


if __name__ == "__main__":
    """Debug harness for running tests with saved outputs."""
    import sys

    # Run pytest on this file
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
