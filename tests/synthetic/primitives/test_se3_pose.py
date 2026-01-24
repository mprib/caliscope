"""Tests for SE3Pose rigid body transformation."""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.synthetic import SE3Pose


class TestConstruction:
    """Test SE3Pose construction and validation."""

    def test_identity_creates_correct_matrices(self):
        """Identity pose has no rotation or translation."""
        pose = SE3Pose.identity()

        assert np.allclose(pose.rotation, np.eye(3))
        assert np.allclose(pose.translation, np.zeros(3))

    def test_valid_rotation_matrix_accepted(self):
        """Proper orthonormal rotation matrices are accepted."""
        # 90-degree rotation around Z-axis
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        translation = np.array([1, 2, 3], dtype=np.float64)

        pose = SE3Pose(rotation=rotation, translation=translation)

        assert np.allclose(pose.rotation, rotation)
        assert np.allclose(pose.translation, translation)

    def test_invalid_rotation_rejected(self):
        """Reject invalid rotation matrices (wrong shape, non-orthogonal, wrong det)."""
        # Non-3x3
        with pytest.raises(ValueError, match="Rotation must be 3x3"):
            SE3Pose(
                rotation=np.eye(2, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
            )

        # Non-orthogonal
        bad_rotation = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="must be orthogonal"):
            SE3Pose(rotation=bad_rotation, translation=np.zeros(3, dtype=np.float64))

        # Wrong determinant (reflection)
        reflection = np.diag([1, 1, -1]).astype(np.float64)
        with pytest.raises(ValueError, match="det=-1.000"):
            SE3Pose(rotation=reflection, translation=np.zeros(3, dtype=np.float64))


class TestFromMatrix:
    """Test SE3Pose.from_matrix() factory method."""

    def test_extracts_rotation_and_translation(self):
        """Extract rotation and translation from homogeneous matrix."""
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        translation = np.array([10, 20, 30], dtype=np.float64)

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation

        pose = SE3Pose.from_matrix(matrix)

        assert np.allclose(pose.rotation, rotation)
        assert np.allclose(pose.translation, translation)

    def test_round_trip_with_matrix_property(self):
        """Converting to matrix and back preserves pose."""
        original = SE3Pose.identity()
        matrix = original.matrix
        reconstructed = SE3Pose.from_matrix(matrix)

        assert np.allclose(reconstructed.rotation, original.rotation)
        assert np.allclose(reconstructed.translation, original.translation)


class TestFromAxisAngle:
    """Test SE3Pose.from_axis_angle() factory method."""

    def test_90_degree_rotation_around_z(self):
        """90-degree rotation around Z-axis."""
        axis = np.array([0, 0, 1], dtype=np.float64)
        angle = np.pi / 2
        translation = np.zeros(3, dtype=np.float64)

        pose = SE3Pose.from_axis_angle(axis, angle, translation)

        # Expected rotation: X->Y, Y->-X
        expected_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

        assert np.allclose(pose.rotation, expected_rotation, atol=1e-10)

    def test_non_unit_axis_rejected(self):
        """Axis must be unit length."""
        non_unit_axis = np.array([1, 1, 1], dtype=np.float64)  # norm = sqrt(3)

        with pytest.raises(ValueError, match="Axis must be unit vector"):
            SE3Pose.from_axis_angle(non_unit_axis, 0.0, np.zeros(3, dtype=np.float64))


class TestLookAt:
    """Test SE3Pose.look_at() factory method."""

    def test_camera_at_positive_x_facing_origin(self):
        """Camera at +X looking at origin."""
        position = np.array([1000, 0, 0], dtype=np.float64)
        target = np.array([0, 0, 0], dtype=np.float64)

        pose = SE3Pose.look_at(position, target)

        # Camera Z-axis should point toward origin (negative X)
        z_axis_world = pose.rotation[2, :]
        expected_forward = np.array([-1, 0, 0], dtype=np.float64)
        assert np.allclose(z_axis_world, expected_forward, atol=1e-10)

        assert np.allclose(pose.translation, position)


class TestCompose:
    """Test SE3Pose.compose() transformation chaining."""

    def test_compose_two_rotations(self):
        """Composing two 90-degree Z-rotations gives 180-degree rotation."""
        # First rotation: 90 degrees around Z
        rot1 = SE3Pose.from_axis_angle(
            axis=np.array([0, 0, 1], dtype=np.float64),
            angle_rad=np.pi / 2,
            translation=np.zeros(3, dtype=np.float64),
        )

        # Second rotation: 90 degrees around Z
        rot2 = SE3Pose.from_axis_angle(
            axis=np.array([0, 0, 1], dtype=np.float64),
            angle_rad=np.pi / 2,
            translation=np.zeros(3, dtype=np.float64),
        )

        # Composed: 180 degrees around Z
        composed = rot1.compose(rot2)

        # Expected: X->-X, Y->-Y, Z->Z
        expected_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

        assert np.allclose(composed.rotation, expected_rotation, atol=1e-10)


class TestInverse:
    """Test SE3Pose.inverse() transformation."""

    def test_pose_compose_inverse_is_identity(self):
        """Composing a pose with its inverse gives identity."""
        # Create non-trivial pose
        pose = SE3Pose.from_axis_angle(
            axis=np.array([1, 1, 1], dtype=np.float64) / np.sqrt(3),
            angle_rad=np.pi / 3,
            translation=np.array([10, 20, 30], dtype=np.float64),
        )

        inv = pose.inverse()
        composed = pose.compose(inv)

        assert np.allclose(composed.rotation, np.eye(3), atol=1e-10)
        assert np.allclose(composed.translation, np.zeros(3), atol=1e-10)


class TestApply:
    """Test SE3Pose.apply() point transformation."""

    def test_identity_leaves_points_unchanged(self):
        """Identity transform doesn't change points."""
        pose = SE3Pose.identity()
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        transformed = pose.apply(points)

        assert np.allclose(transformed, points)

    def test_rotation_and_translation(self):
        """Combined rotation and translation."""
        # Rotate 90 degrees around Z, then translate
        pose = SE3Pose.from_axis_angle(
            axis=np.array([0, 0, 1], dtype=np.float64),
            angle_rad=np.pi / 2,
            translation=np.array([5, 10, 0], dtype=np.float64),
        )
        points = np.array([[1, 0, 0]], dtype=np.float64)

        transformed = pose.apply(points)

        # (1, 0, 0) rotates to (0, 1, 0), then translates to (5, 11, 0)
        expected = np.array([[5, 11, 0]], dtype=np.float64)
        assert np.allclose(transformed, expected, atol=1e-10)


if __name__ == "__main__":
    """Debug harness for running tests with saved outputs."""
    import sys

    # Run pytest on this file
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
