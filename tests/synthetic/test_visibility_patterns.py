"""
Unit tests for visibility mask factories.

Tests verify shape, dtype, and key invariants for each pattern:
- full_visibility: All True baseline
- disconnected_components: Disjoint camera groups
- sequential_overlap: Chain-linked visibility
- partial_visibility: Random occlusion with minimum coverage
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.synthetic.visibility_patterns import (
    disconnected_components,
    full_visibility,
    partial_visibility,
    sequential_overlap,
)

# Small test parameters for fast execution
N_CAMERAS = 4
N_FRAMES = 20
N_POINTS = 35


def test_full_visibility():
    """All points visible from all cameras at all frames."""
    mask = full_visibility(N_CAMERAS, N_FRAMES, N_POINTS)

    # Verify shape
    assert mask.shape == (N_CAMERAS, N_FRAMES, N_POINTS)

    # Verify dtype
    assert mask.dtype == np.bool_

    # Verify all True
    assert np.all(mask)


def test_disconnected_components_valid():
    """Components have no shared observations."""
    component_sizes = [2, 2]  # Two components of 2 cameras each
    mask = disconnected_components(N_CAMERAS, N_FRAMES, N_POINTS, component_sizes)

    # Verify shape and dtype
    assert mask.shape == (N_CAMERAS, N_FRAMES, N_POINTS)
    assert mask.dtype == np.bool_

    # Component 1: cameras 0, 1
    # Component 2: cameras 2, 3
    # Verify no shared frames between components
    comp1_frames = np.any(mask[:2, :, :], axis=(0, 2))  # Frames visible to comp1
    comp2_frames = np.any(mask[2:, :, :], axis=(0, 2))  # Frames visible to comp2

    # Components should have no shared True values
    shared_frames = np.logical_and(comp1_frames, comp2_frames)
    assert not np.any(shared_frames), "Components should not share any frames"


def test_disconnected_components_frame_coverage():
    """Each component sees its allocated frames fully."""
    component_sizes = [2, 2]
    mask = disconnected_components(N_CAMERAS, N_FRAMES, N_POINTS, component_sizes)

    n_components = len(component_sizes)
    camera_idx = 0

    for comp_idx, comp_size in enumerate(component_sizes):
        # Calculate expected frame range for this component
        frame_start = (comp_idx * N_FRAMES) // n_components
        frame_end = ((comp_idx + 1) * N_FRAMES) // n_components

        # Verify all cameras in component see their assigned frames
        for _ in range(comp_size):
            # Within frame range: should be all True
            assert np.all(mask[camera_idx, frame_start:frame_end, :])

            # Outside frame range: should be all False
            if frame_start > 0:
                assert not np.any(mask[camera_idx, :frame_start, :])
            if frame_end < N_FRAMES:
                assert not np.any(mask[camera_idx, frame_end:, :])

            camera_idx += 1


def test_disconnected_components_invalid_sizes():
    """Raises ValueError if component_sizes don't sum to n_cameras."""
    with pytest.raises(ValueError, match="component_sizes must sum to n_cameras"):
        disconnected_components(N_CAMERAS, N_FRAMES, N_POINTS, component_sizes=[2, 1])

    with pytest.raises(ValueError, match="component_sizes must sum to n_cameras"):
        disconnected_components(N_CAMERAS, N_FRAMES, N_POINTS, component_sizes=[5])


def test_sequential_overlap_valid():
    """Adjacent cameras share exactly overlap_frames."""
    overlap_frames = 5
    mask = sequential_overlap(N_CAMERAS, N_FRAMES, N_POINTS, overlap_frames)

    # Verify shape and dtype
    assert mask.shape == (N_CAMERAS, N_FRAMES, N_POINTS)
    assert mask.dtype == np.bool_

    # Check each adjacent pair
    for cam_idx in range(N_CAMERAS - 1):
        # Get frames visible to each camera (reduce over points dimension)
        cam_i_frames = np.any(mask[cam_idx, :, :], axis=1)
        cam_j_frames = np.any(mask[cam_idx + 1, :, :], axis=1)

        # Count shared frames
        shared_frames = np.logical_and(cam_i_frames, cam_j_frames)
        shared_count = np.sum(shared_frames)

        # Adjacent cameras should share exactly overlap_frames
        assert shared_count == overlap_frames, (
            f"Cameras {cam_idx} and {cam_idx + 1} share {shared_count} frames, expected {overlap_frames}"
        )


def test_sequential_overlap_non_adjacent_disjoint():
    """Non-adjacent cameras share no frames."""
    # Need more frames to guarantee non-adjacent disjointness
    # With overlap=5, n_cameras=4, need frames >= overlap*(n_cameras+1) = 25
    n_frames = 30
    overlap_frames = 5
    mask = sequential_overlap(N_CAMERAS, n_frames, N_POINTS, overlap_frames)

    # Check non-adjacent pairs (gap of at least 1)
    for cam_i in range(N_CAMERAS):
        for cam_j in range(cam_i + 2, N_CAMERAS):
            # Get frames visible to each camera
            cam_i_frames = np.any(mask[cam_i, :, :], axis=1)
            cam_j_frames = np.any(mask[cam_j, :, :], axis=1)

            # Count shared frames
            shared_frames = np.logical_and(cam_i_frames, cam_j_frames)
            shared_count = np.sum(shared_frames)

            # Non-adjacent cameras should share no frames
            assert shared_count == 0, (
                f"Non-adjacent cameras {cam_i} and {cam_j} share {shared_count} frames, expected 0"
            )


def test_sequential_overlap_insufficient_frames():
    """Raises ValueError if insufficient frames for chain."""
    overlap_frames = 10
    min_required = overlap_frames * (N_CAMERAS - 1) + 1  # 31 frames

    # Should succeed with sufficient frames
    mask = sequential_overlap(N_CAMERAS, min_required, N_POINTS, overlap_frames)
    assert mask.shape == (N_CAMERAS, min_required, N_POINTS)

    # Should fail with insufficient frames
    with pytest.raises(ValueError, match="n_sync_indices .* is insufficient"):
        sequential_overlap(N_CAMERAS, min_required - 1, N_POINTS, overlap_frames)


def test_partial_visibility_shape_dtype():
    """Verify basic properties of partial visibility mask."""
    visibility_fraction = 0.6
    rng = np.random.default_rng(seed=42)

    mask = partial_visibility(N_CAMERAS, N_FRAMES, N_POINTS, visibility_fraction, rng)

    # Verify shape and dtype
    assert mask.shape == (N_CAMERAS, N_FRAMES, N_POINTS)
    assert mask.dtype == np.bool_


def test_partial_visibility_minimum_coverage():
    """Every point visible to at least 2 cameras (triangulation requirement)."""
    visibility_fraction = 0.3  # Low fraction to stress-test minimum coverage
    rng = np.random.default_rng(seed=42)

    mask = partial_visibility(N_CAMERAS, N_FRAMES, N_POINTS, visibility_fraction, rng)

    # Check each (sync_index, point_id) combination
    for sync_idx in range(N_FRAMES):
        for point_idx in range(N_POINTS):
            visible_count = np.sum(mask[:, sync_idx, point_idx])
            assert visible_count >= 2, (
                f"Point ({sync_idx}, {point_idx}) visible to only {visible_count} cameras, need >= 2"
            )


def test_partial_visibility_reproducible():
    """Same seed produces same mask."""
    visibility_fraction = 0.5

    # Generate mask twice with same seed
    mask1 = partial_visibility(N_CAMERAS, N_FRAMES, N_POINTS, visibility_fraction, np.random.default_rng(seed=123))
    mask2 = partial_visibility(N_CAMERAS, N_FRAMES, N_POINTS, visibility_fraction, np.random.default_rng(seed=123))

    assert np.array_equal(mask1, mask2), "Same seed should produce identical masks"

    # Different seed should produce different mask
    mask3 = partial_visibility(N_CAMERAS, N_FRAMES, N_POINTS, visibility_fraction, np.random.default_rng(seed=456))

    assert not np.array_equal(mask1, mask3), "Different seeds should produce different masks"


def test_partial_visibility_respects_fraction():
    """Visibility fraction approximately controls overall density."""
    visibility_fraction = 0.7
    rng = np.random.default_rng(seed=42)

    mask = partial_visibility(N_CAMERAS, N_FRAMES, N_POINTS, visibility_fraction, rng)

    # Calculate actual visibility fraction
    # Note: May be slightly higher due to minimum coverage guarantee
    actual_fraction = np.sum(mask) / mask.size

    # Should be at least the requested fraction (guaranteed minimum coverage may increase it)
    assert actual_fraction >= visibility_fraction * 0.9, (
        f"Actual fraction {actual_fraction:.2f} significantly below requested {visibility_fraction:.2f}"
    )


if __name__ == "__main__":
    """Debug harness for visual inspection and breakpoint debugging."""
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Running visibility pattern tests...")
    print(f"Debug outputs will be saved to: {debug_dir}")

    # Run all tests
    test_full_visibility()
    print("✓ test_full_visibility")

    test_disconnected_components_valid()
    print("✓ test_disconnected_components_valid")

    test_disconnected_components_frame_coverage()
    print("✓ test_disconnected_components_frame_coverage")

    test_disconnected_components_invalid_sizes()
    print("✓ test_disconnected_components_invalid_sizes")

    test_sequential_overlap_valid()
    print("✓ test_sequential_overlap_valid")

    test_sequential_overlap_non_adjacent_disjoint()
    print("✓ test_sequential_overlap_non_adjacent_disjoint")

    test_sequential_overlap_insufficient_frames()
    print("✓ test_sequential_overlap_insufficient_frames")

    test_partial_visibility_shape_dtype()
    print("✓ test_partial_visibility_shape_dtype")

    test_partial_visibility_minimum_coverage()
    print("✓ test_partial_visibility_minimum_coverage")

    test_partial_visibility_reproducible()
    print("✓ test_partial_visibility_reproducible")

    test_partial_visibility_respects_fraction()
    print("✓ test_partial_visibility_respects_fraction")

    print("\nAll tests passed!")

    # Generate sample masks for visual inspection
    print("\nGenerating sample masks for inspection...")

    rng = np.random.default_rng(seed=42)

    # Full visibility
    full_mask = full_visibility(N_CAMERAS, N_FRAMES, N_POINTS)
    np.save(debug_dir / "full_visibility.npy", full_mask)

    # Disconnected components
    disconnected_mask = disconnected_components(N_CAMERAS, N_FRAMES, N_POINTS, [2, 2])
    np.save(debug_dir / "disconnected_components.npy", disconnected_mask)

    # Sequential overlap
    sequential_mask = sequential_overlap(N_CAMERAS, N_FRAMES, N_POINTS, overlap_frames=5)
    np.save(debug_dir / "sequential_overlap.npy", sequential_mask)

    # Partial visibility
    partial_mask = partial_visibility(N_CAMERAS, N_FRAMES, N_POINTS, 0.6, rng)
    np.save(debug_dir / "partial_visibility.npy", partial_mask)

    print(f"Sample masks saved to {debug_dir}")
