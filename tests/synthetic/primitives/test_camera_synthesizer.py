"""Tests for CameraSynthesizer fluent builder."""

import numpy as np
import pytest

from caliscope.synthetic.camera_synthesizer import CameraSynthesizer, strip_extrinsics
from caliscope.synthetic.se3_pose import SE3Pose


class TestCameraSynthesizer:
    """Tests for CameraSynthesizer camera array builder."""

    def test_single_ring_creates_correct_camera_count(self) -> None:
        array = CameraSynthesizer().add_ring(n=4, radius_mm=1000).build()
        assert len(array.cameras) == 4

    def test_ports_are_sequential(self) -> None:
        array = CameraSynthesizer().add_ring(n=4, radius_mm=1000).build()
        assert list(array.cameras.keys()) == [0, 1, 2, 3]

    def test_two_rings_accumulate_ports(self) -> None:
        array = (
            CameraSynthesizer()
            .add_ring(n=4, radius_mm=2000, height_mm=0)
            .add_ring(n=3, radius_mm=2000, height_mm=500)
            .build()
        )
        assert len(array.cameras) == 7
        assert list(array.cameras.keys()) == [0, 1, 2, 3, 4, 5, 6]

    def test_drop_ports_creates_gaps(self) -> None:
        array = CameraSynthesizer().add_ring(n=4, radius_mm=2000).drop_ports(1, 3).build()
        assert list(array.cameras.keys()) == [0, 2]

    def test_angular_offset_rotates_ring(self) -> None:
        """45-degree offset should change camera positions."""
        array_no_offset = CameraSynthesizer().add_ring(n=4, radius_mm=1000).build()
        array_with_offset = CameraSynthesizer().add_ring(n=4, radius_mm=1000, angular_offset_deg=45).build()

        # Camera 0 position should differ
        pos0_no_offset = -array_no_offset.cameras[0].rotation.T @ array_no_offset.cameras[0].translation  # type: ignore[union-attr]
        pos0_with_offset = -array_with_offset.cameras[0].rotation.T @ array_with_offset.cameras[0].translation  # type: ignore[union-attr]

        assert not np.allclose(pos0_no_offset, pos0_with_offset, atol=1)

    def test_rejects_insufficient_cameras(self) -> None:
        """Must have at least 2 cameras after drops."""
        with pytest.raises(ValueError, match="at least 2 cameras"):
            CameraSynthesizer().add_ring(n=2, radius_mm=1000).drop_ports(0, 1).build()

    def test_roll_variation_applies_random_roll(self) -> None:
        """With roll variation, cameras should have different orientations."""
        # Same seed should give reproducible results
        array1 = CameraSynthesizer().add_ring(n=4, radius_mm=1000, roll_variation_deg=10, random_seed=42).build()
        array2 = CameraSynthesizer().add_ring(n=4, radius_mm=1000, roll_variation_deg=10, random_seed=42).build()

        # Same seed = same rotations
        for port in array1.cameras:
            np.testing.assert_allclose(
                array1.cameras[port].rotation,  # type: ignore[arg-type]
                array2.cameras[port].rotation,  # type: ignore[arg-type]
            )

    def test_different_seeds_give_different_results(self) -> None:
        array1 = CameraSynthesizer().add_ring(n=4, radius_mm=1000, roll_variation_deg=10, random_seed=42).build()
        array2 = CameraSynthesizer().add_ring(n=4, radius_mm=1000, roll_variation_deg=10, random_seed=99).build()

        # Different seeds should give different rotations
        assert not np.allclose(
            array1.cameras[0].rotation,  # type: ignore[arg-type]
            array2.cameras[0].rotation,  # type: ignore[arg-type]
        )


class TestAddLine:
    """Tests for add_line camera arrangement."""

    def test_creates_correct_camera_count(self) -> None:
        array = CameraSynthesizer().add_line(n=5, spacing_mm=200).build()
        assert len(array.cameras) == 5

    def test_cameras_are_centered(self) -> None:
        """Line should be centered around x=0."""
        array = CameraSynthesizer().add_line(n=3, spacing_mm=100).build()

        # Extract positions from extrinsics
        positions = []
        for port, cam in array.cameras.items():
            pos = -cam.rotation.T @ cam.translation  # type: ignore[union-attr]
            positions.append(pos)

        x_coords = [p[0] for p in positions]
        # Should be centered: -100, 0, 100
        assert np.isclose(sum(x_coords), 0, atol=1)


class TestStripExtrinsics:
    """Tests for strip_extrinsics utility function."""

    def test_removes_extrinsics(self) -> None:
        original = CameraSynthesizer().add_ring(n=4, radius_mm=1000).build()
        stripped = strip_extrinsics(original)

        for port, cam in stripped.cameras.items():
            assert cam.rotation is None
            assert cam.translation is None

    def test_preserves_intrinsics(self) -> None:
        original = CameraSynthesizer().add_ring(n=4, radius_mm=1000).build()
        stripped = strip_extrinsics(original)

        for port, cam in stripped.cameras.items():
            orig_cam = original.cameras[port]
            assert cam.matrix is not None
            assert orig_cam.matrix is not None
            np.testing.assert_allclose(cam.matrix, orig_cam.matrix)
            assert cam.distortions is not None
            assert orig_cam.distortions is not None
            np.testing.assert_allclose(cam.distortions, orig_cam.distortions)
            assert cam.size == orig_cam.size

    def test_does_not_modify_original(self) -> None:
        original = CameraSynthesizer().add_ring(n=4, radius_mm=1000).build()
        original_rotation = original.cameras[0].rotation.copy()  # type: ignore[union-attr]
        original_translation = original.cameras[0].translation.copy()  # type: ignore[union-attr]

        strip_extrinsics(original)

        # Original should still have extrinsics
        assert original.cameras[0].rotation is not None
        assert original.cameras[0].translation is not None
        np.testing.assert_allclose(original.cameras[0].rotation, original_rotation)
        np.testing.assert_allclose(original.cameras[0].translation, original_translation)


class TestSE3PoseRotations:
    """Tests for SE3Pose.with_roll and with_pitch methods."""

    def test_pitch_90_degrees_looks_straight_up(self) -> None:
        """CV engineer's recommended verification test."""
        # Camera at +X looking toward origin
        pose = SE3Pose.look_at(
            position=np.array([1000.0, 0.0, 0.0]),
            target=np.array([0.0, 0.0, 0.0]),
        )
        # Original camera Z points toward -X world

        pitched = pose.with_pitch(np.pi / 2)

        # After 90° pitch up: camera Z should point toward +Z world
        camera_z_in_world = pitched.rotation[2]  # Third row
        np.testing.assert_allclose(camera_z_in_world, [0, 0, 1], atol=1e-10)

    def test_roll_preserves_optical_axis(self) -> None:
        """Roll should rotate around Z, not change where camera points."""
        pose = SE3Pose.look_at(
            position=np.array([1000.0, 0.0, 0.0]),
            target=np.array([0.0, 0.0, 0.0]),
        )
        original_z = pose.rotation[2].copy()

        rolled = pose.with_roll(np.pi / 4)

        # Camera Z (forward direction) should be unchanged
        np.testing.assert_allclose(rolled.rotation[2], original_z, atol=1e-10)

    def test_small_variations_stay_close_to_original(self) -> None:
        """Small roll/pitch should produce rotation close to identity delta."""
        pose = SE3Pose.look_at(
            position=np.array([0.0, 1000.0, 500.0]),
            target=np.array([0.0, 0.0, 0.0]),
        )

        # 5 degrees is small
        varied = pose.with_pitch(np.radians(5)).with_roll(np.radians(5))

        # Rotation matrices should be similar (Frobenius norm of difference)
        diff_norm = np.linalg.norm(varied.rotation - pose.rotation, "fro")
        assert diff_norm < 0.2  # Small rotation = small matrix difference


if __name__ == "__main__":
    from pathlib import Path

    # Debug harness for visual inspection
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Creating test camera arrays...")

    # Basic ring
    ring = CameraSynthesizer().add_ring(n=4, radius_mm=1000, height_mm=500).build()
    print(f"Ring: {len(ring.cameras)} cameras at ports {list(ring.cameras.keys())}")

    # Two staggered rings
    double_ring = (
        CameraSynthesizer()
        .add_ring(n=4, radius_mm=2000, height_mm=0)
        .add_ring(n=4, radius_mm=2000, height_mm=500, angular_offset_deg=45)
        .build()
    )
    print(f"Double ring: {len(double_ring.cameras)} cameras")

    # With drops
    sparse = CameraSynthesizer().add_ring(n=6, radius_mm=1500).drop_ports(1, 4).build()
    print(f"Sparse: {len(sparse.cameras)} cameras at ports {list(sparse.cameras.keys())}")

    # Line with curvature
    line = CameraSynthesizer().add_line(n=5, spacing_mm=300, curvature=1.0).build()
    print(f"Curved line: {len(line.cameras)} cameras")

    # Run tests
    pytest.main([__file__, "-v"])
