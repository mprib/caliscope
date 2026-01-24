"""Tests for camera rig generator functions."""

import numpy as np
import pytest

from caliscope.synthetic.camera_rigs import (
    linear_rig,
    nested_rings_rig,
    ring_rig,
    strip_extrinsics,
)


class TestRingRig:
    """Tests for ring_rig camera arrangement."""

    def test_creates_correct_number_of_cameras(self) -> None:
        rig = ring_rig(n_cameras=4, radius_mm=1000)
        assert len(rig.cameras) == 4

    def test_rejects_invalid_inputs(self) -> None:
        """Reject invalid camera count or radius."""
        with pytest.raises(ValueError, match="at least 2 cameras"):
            ring_rig(n_cameras=1, radius_mm=1000)

        with pytest.raises(ValueError, match="must be positive"):
            ring_rig(n_cameras=4, radius_mm=0)


class TestLinearRig:
    """Tests for linear_rig camera arrangement."""

    def test_creates_correct_number_of_cameras(self) -> None:
        rig = linear_rig(n_cameras=5, spacing_mm=200)
        assert len(rig.cameras) == 5

    def test_rejects_invalid_inputs(self) -> None:
        """Reject invalid camera count or spacing."""
        with pytest.raises(ValueError, match="at least 2 cameras"):
            linear_rig(n_cameras=1, spacing_mm=100)

        with pytest.raises(ValueError, match="must be positive"):
            linear_rig(n_cameras=4, spacing_mm=0)


class TestNestedRingsRig:
    """Tests for nested_rings_rig camera arrangement."""

    def test_creates_correct_number_of_cameras(self) -> None:
        rig = nested_rings_rig(inner_n=3, outer_n=4, inner_radius_mm=500, outer_radius_mm=1500)
        assert len(rig.cameras) == 7  # 3 + 4

    def test_rejects_invalid_inputs(self) -> None:
        """Reject invalid camera counts or radii."""
        with pytest.raises(ValueError, match="at least 2 cameras"):
            nested_rings_rig(inner_n=1, outer_n=4, inner_radius_mm=500, outer_radius_mm=1500)

        with pytest.raises(ValueError, match="Inner radius must be < outer radius"):
            nested_rings_rig(inner_n=3, outer_n=4, inner_radius_mm=1500, outer_radius_mm=1000)


class TestStripExtrinsics:
    """Tests for strip_extrinsics utility function."""

    def test_removes_extrinsics(self) -> None:
        original = ring_rig(n_cameras=4, radius_mm=1000)
        stripped = strip_extrinsics(original)

        for port, cam in stripped.cameras.items():
            assert cam.rotation is None
            assert cam.translation is None

    def test_preserves_intrinsics(self) -> None:
        original = ring_rig(n_cameras=4, radius_mm=1000)
        stripped = strip_extrinsics(original)

        for port, cam in stripped.cameras.items():
            orig_cam = original.cameras[port]
            assert cam.matrix is not None
            assert orig_cam.matrix is not None
            assert np.allclose(cam.matrix, orig_cam.matrix)
            assert cam.distortions is not None
            assert orig_cam.distortions is not None
            assert np.allclose(cam.distortions, orig_cam.distortions)
            assert cam.size == orig_cam.size

    def test_does_not_modify_original(self) -> None:
        original = ring_rig(n_cameras=4, radius_mm=1000)
        original_rotation = original.cameras[0].rotation.copy()  # type: ignore
        original_translation = original.cameras[0].translation.copy()  # type: ignore

        strip_extrinsics(original)

        # Original should still have extrinsics
        assert original.cameras[0].rotation is not None
        assert original.cameras[0].translation is not None
        assert np.allclose(original.cameras[0].rotation, original_rotation)
        assert np.allclose(original.cameras[0].translation, original_translation)


if __name__ == "__main__":
    from pathlib import Path

    # Debug harness for visual inspection
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Create test rigs for visual inspection
    print("Creating test rigs...")

    ring = ring_rig(n_cameras=4, radius_mm=1000, height_mm=500)
    print(f"Ring rig: {len(ring.cameras)} cameras")

    linear = linear_rig(n_cameras=5, spacing_mm=300, curvature=1.0)
    print(f"Linear rig: {len(linear.cameras)} cameras")

    nested = nested_rings_rig(inner_n=3, outer_n=6, inner_radius_mm=500, outer_radius_mm=1500)
    print(f"Nested rings rig: {len(nested.cameras)} cameras")

    # Run tests
    pytest.main([__file__, "-v"])
