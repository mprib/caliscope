"""
Visibility mask factories for synthetic calibration testing.

A visibility mask is a 3D boolean array: (n_cameras, n_sync_indices, n_points_per_frame)
where True means the point is visible to that camera at that frame.

Standard test scenarios:
- full_visibility: All points visible from all cameras at all frames (baseline)
- disconnected_components: Camera groups with no shared observations
- sequential_overlap: Chain of cameras, each sees only neighbors
- partial_visibility: Random occlusion with guaranteed minimum coverage
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def full_visibility(
    n_cameras: int,
    n_sync_indices: int,
    n_points_per_frame: int,
) -> NDArray[np.bool_]:
    """
    All points visible from all cameras at all frames.

    This is the default behavior when no mask is provided.
    Useful as a baseline or for explicit "no occlusion" tests.

    Returns:
        Boolean mask of shape (n_cameras, n_sync_indices, n_points_per_frame), all True.
    """
    return np.ones((n_cameras, n_sync_indices, n_points_per_frame), dtype=np.bool_)


def disconnected_components(
    n_cameras: int,
    n_sync_indices: int,
    n_points_per_frame: int,
    component_sizes: list[int],
) -> NDArray[np.bool_]:
    """
    Create visibility pattern with disconnected camera groups.

    Each component sees a disjoint subset of frames. No shared observations
    between components means the visibility graph is disconnected.

    Example: 4 cameras, component_sizes=[2, 2]
    - Cameras 0, 1 see frames 0 to n//2
    - Cameras 2, 3 see frames n//2 to n
    - No frame is seen by cameras from both components

    Args:
        n_cameras: Total number of cameras
        n_sync_indices: Number of temporal frames
        n_points_per_frame: Points per frame (e.g., grid_rows * grid_cols)
        component_sizes: List of camera counts per component.
                        Must sum to n_cameras.

    Returns:
        Boolean mask of shape (n_cameras, n_sync_indices, n_points_per_frame).

    Raises:
        ValueError: If component_sizes doesn't sum to n_cameras.
    """
    if sum(component_sizes) != n_cameras:
        raise ValueError(f"component_sizes must sum to n_cameras ({n_cameras}), got {sum(component_sizes)}")

    mask = np.zeros((n_cameras, n_sync_indices, n_points_per_frame), dtype=np.bool_)

    n_components = len(component_sizes)
    camera_idx = 0

    for comp_idx, comp_size in enumerate(component_sizes):
        # Divide frames proportionally across components
        # Each component gets a contiguous block of frames
        frame_start = (comp_idx * n_sync_indices) // n_components
        frame_end = ((comp_idx + 1) * n_sync_indices) // n_components

        # All cameras in this component see their assigned frame range
        for _ in range(comp_size):
            mask[camera_idx, frame_start:frame_end, :] = True
            camera_idx += 1

    return mask


def sequential_overlap(
    n_cameras: int,
    n_sync_indices: int,
    n_points_per_frame: int,
    overlap_frames: int,
) -> NDArray[np.bool_]:
    """
    Create chain-linked visibility pattern.

    Camera i and camera i+1 share exactly `overlap_frames` frames.
    Non-adjacent cameras share no frames.

    This tests error accumulation along weak links.

    Example: 4 cameras, 30 frames, overlap_frames=5
    - Camera 0: frames 0-9
    - Camera 1: frames 5-14 (overlaps 5-9 with cam 0)
    - Camera 2: frames 10-19 (overlaps 10-14 with cam 1)
    - Camera 3: frames 15-24 (overlaps 15-19 with cam 2)

    Args:
        n_cameras: Number of cameras in the chain
        n_sync_indices: Total number of temporal frames
        n_points_per_frame: Points per frame
        overlap_frames: Frames shared between adjacent cameras

    Returns:
        Boolean mask of shape (n_cameras, n_sync_indices, n_points_per_frame).

    Raises:
        ValueError: If n_sync_indices is insufficient for the chain length.
    """
    # Calculate required frames for the chain
    # Each camera needs its own window, but overlaps reduce total requirement
    # Window size per camera: (n_sync_indices + (n_cameras - 1) * overlap_frames) / n_cameras
    # Simplified: total frames needed >= overlap_frames * (n_cameras - 1) + 1
    min_frames = overlap_frames * (n_cameras - 1) + 1
    if n_sync_indices < min_frames:
        raise ValueError(
            f"n_sync_indices ({n_sync_indices}) is insufficient for chain with "
            f"{n_cameras} cameras and {overlap_frames} overlap frames. "
            f"Minimum required: {min_frames}"
        )

    mask = np.zeros((n_cameras, n_sync_indices, n_points_per_frame), dtype=np.bool_)

    # Calculate window size for each camera
    # We want each camera to see approximately the same number of frames
    # with overlap_frames shared between adjacent cameras

    # Total "coverage" if we lay out windows with overlaps:
    # window_size * n_cameras - overlap_frames * (n_cameras - 1) = n_sync_indices
    # Solving: window_size = (n_sync_indices + overlap_frames * (n_cameras - 1)) / n_cameras
    window_size = (n_sync_indices + overlap_frames * (n_cameras - 1)) // n_cameras

    for cam_idx in range(n_cameras):
        # Each camera's window starts where the previous camera's overlap begins
        # Camera i starts at: i * (window_size - overlap_frames)
        start_frame = cam_idx * (window_size - overlap_frames)
        end_frame = min(start_frame + window_size, n_sync_indices)

        mask[cam_idx, start_frame:end_frame, :] = True

    return mask


def partial_visibility(
    n_cameras: int,
    n_sync_indices: int,
    n_points_per_frame: int,
    visibility_fraction: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """
    Random partial visibility with guaranteed minimum coverage.

    Each point is visible from a random subset of cameras.
    Guarantees each point is seen by at least 2 cameras (triangulation minimum).

    Args:
        n_cameras: Number of cameras
        n_sync_indices: Number of temporal frames
        n_points_per_frame: Points per frame
        visibility_fraction: Fraction of camera-point pairs that are visible (0.0 to 1.0).
                            Clamped to ensure at least 2 cameras per point.
        rng: Random generator for reproducibility

    Returns:
        Boolean mask of shape (n_cameras, n_sync_indices, n_points_per_frame).
    """

    # Start with random visibility based on the requested fraction
    mask = rng.random((n_cameras, n_sync_indices, n_points_per_frame)) < visibility_fraction

    # Ensure each point is visible from at least 2 cameras
    # Check each (sync_index, point_id) combination
    for sync_idx in range(n_sync_indices):
        for point_idx in range(n_points_per_frame):
            visible_count = np.sum(mask[:, sync_idx, point_idx])

            # If fewer than 2 cameras see this point, randomly enable cameras until we have 2
            if visible_count < 2:
                # Find cameras that don't currently see the point
                hidden_cameras = np.where(~mask[:, sync_idx, point_idx])[0]

                # Randomly select cameras to enable
                n_to_enable = 2 - visible_count
                if len(hidden_cameras) >= n_to_enable:
                    cameras_to_enable = rng.choice(hidden_cameras, size=n_to_enable, replace=False)
                else:
                    # If we don't have enough hidden cameras, enable all of them
                    cameras_to_enable = hidden_cameras

                mask[cameras_to_enable, sync_idx, point_idx] = True

    return mask
