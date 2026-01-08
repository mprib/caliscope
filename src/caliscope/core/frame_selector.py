"""Automatic frame selection for intrinsic camera calibration.

Implements a deterministic, coverage-optimizing greedy algorithm for selecting
calibration frames from tracked charuco corner data. The algorithm maximizes
spatial coverage across the image while considering edge/corner regions
(critical for distortion estimation) and pose diversity.

Literature foundation:
- Zhang (2000): Different orientations needed for parameter observability
- MATLAB/OpenCV docs: Peripheral coverage constrains distortion model
- Krause (2012): Greedy submodular maximization achieves (1-1/e) approximation

Note: For charuco boards (planar targets), obj_loc_z is always 0. Pose diversity
is computed from image-space features (centroid, spread) rather than 3D pose.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from caliscope.core.point_data import ImagePoints


@dataclass(frozen=True)
class FrameSelectionResult:
    """Immutable result of frame selection for intrinsic calibration."""

    selected_frames: list[int]  # sync_index values

    # Quality metrics (for diagnostics/logging)
    coverage_fraction: float  # Target > 0.80 (20+ of 25 cells)
    edge_coverage_fraction: float  # Target > 0.75
    corner_coverage_fraction: float  # Target > 0.50
    pose_diversity: float  # Variance of pose features

    # Selection metadata
    eligible_frame_count: int
    total_frame_count: int


def select_calibration_frames(
    image_points: ImagePoints,
    port: int,
    image_size: tuple[int, int],
    *,
    target_frame_count: int = 30,
    min_corners_per_frame: int = 6,
    min_coverage_fraction: float = 0.10,
    grid_size: int = 5,
) -> FrameSelectionResult:
    """Select optimal frames for intrinsic camera calibration.

    Uses a deterministic, coverage-optimizing greedy algorithm that:
    1. Filters frames by corner count and coverage criteria
    2. Greedily selects frames to maximize image grid coverage
    3. Weights edge and corner regions for distortion estimation
    4. Considers pose diversity for robust parameter estimation

    Determinism: Ties are broken by sync_index (lowest first). No random sampling.

    Args:
        image_points: Detected charuco corners across all frames
        port: Camera port to select frames for
        image_size: (width, height) of the camera image
        target_frame_count: Maximum frames to select (default 30)
        min_corners_per_frame: Minimum corners required per frame (default 6)
        min_coverage_fraction: Minimum image coverage per frame (default 0.10)
        grid_size: Coverage grid dimension (default 5 for 5x5 = 25 cells)

    Returns:
        FrameSelectionResult with selected sync_index values and quality metrics
    """
    # Filter to specified port
    port_df = cast(pd.DataFrame, image_points.df[image_points.df["port"] == port].copy())
    total_frame_count = int(port_df["sync_index"].nunique())

    if total_frame_count == 0:
        return FrameSelectionResult(
            selected_frames=[],
            coverage_fraction=0.0,
            edge_coverage_fraction=0.0,
            corner_coverage_fraction=0.0,
            pose_diversity=0.0,
            eligible_frame_count=0,
            total_frame_count=0,
        )

    # Filter eligible frames
    eligible_frames = _filter_eligible_frames(port_df, image_size, min_corners_per_frame, min_coverage_fraction)

    if not eligible_frames:
        return FrameSelectionResult(
            selected_frames=[],
            coverage_fraction=0.0,
            edge_coverage_fraction=0.0,
            corner_coverage_fraction=0.0,
            pose_diversity=0.0,
            eligible_frame_count=0,
            total_frame_count=total_frame_count,
        )

    # Precompute coverage and pose features for all eligible frames
    frame_data: dict[int, tuple[set[tuple[int, int]], np.ndarray]] = {}
    for sync_index in eligible_frames:
        frame_df = cast(pd.DataFrame, port_df[port_df["sync_index"] == sync_index])
        coverage = _compute_frame_coverage(frame_df, image_size, grid_size)
        pose = _compute_pose_features(frame_df, image_size)
        frame_data[sync_index] = (coverage, pose)

    # Greedy selection
    selected_frames = _greedy_select(frame_data, target_frame_count, grid_size, min_score=0.01)

    # Compute quality metrics
    metrics = _compute_quality_metrics(frame_data, selected_frames, grid_size)

    return FrameSelectionResult(
        selected_frames=selected_frames,
        coverage_fraction=metrics["coverage_fraction"],
        edge_coverage_fraction=metrics["edge_coverage_fraction"],
        corner_coverage_fraction=metrics["corner_coverage_fraction"],
        pose_diversity=metrics["pose_diversity"],
        eligible_frame_count=len(eligible_frames),
        total_frame_count=total_frame_count,
    )


def _compute_board_aspect_ratio(port_df: pd.DataFrame) -> float:
    """Infer board aspect ratio from obj_loc spread across all frames.

    The obj_loc values represent physical board coordinates, so their spread
    reveals the board's physical aspect ratio (width/height).
    """

    obj_x_range = float(port_df["obj_loc_x"].max() - port_df["obj_loc_x"].min())
    obj_y_range = float(port_df["obj_loc_y"].max() - port_df["obj_loc_y"].min())

    if obj_y_range < 1e-6:
        return 1.0
    return obj_x_range / obj_y_range


def _max_possible_bbox_area(image_size: tuple[int, int], board_aspect_ratio: float) -> float:
    """Calculate max bbox area if board filled frame while preserving aspect ratio.

    This is the theoretical maximum - if the board's corners spanned the full
    image while maintaining the board's physical aspect ratio.
    """
    width, height = image_size
    image_aspect = width / height

    if board_aspect_ratio > image_aspect:
        # Board wider than image aspect - width-constrained
        max_bbox_width = float(width)
        max_bbox_height = width / board_aspect_ratio
    else:
        # Board taller than image aspect - height-constrained
        max_bbox_height = float(height)
        max_bbox_width = height * board_aspect_ratio

    return max_bbox_width * max_bbox_height


def _filter_eligible_frames(
    port_df: pd.DataFrame,
    image_size: tuple[int, int],
    min_corners: int,
    min_coverage: float,
) -> list[int]:
    """Filter frames meeting minimum corner count and relative coverage criteria.

    Coverage is computed relative to the maximum possible bbox for the board's
    aspect ratio, not the full image area. This handles small boards (e.g., 3x4
    charuco with only 6 corners) correctly - a board whose corners fill 30% of
    the frame should pass a 10% threshold even if its corner bbox is only 3% of
    image area.

    Note: For ChArUco boards, internal corners are inset from the board edge by
    one square width, so a denser grid has corners closer to the true edge.
    Classic checkerboard patterns would differ.
    """
    # Compute board aspect ratio from obj_loc spread across all frames
    board_aspect = _compute_board_aspect_ratio(port_df)
    max_bbox = _max_possible_bbox_area(image_size, board_aspect)

    eligible: list[int] = []
    grouped = port_df.groupby("sync_index")
    for sync_index_key, frame_group in grouped:
        frame_df = cast(pd.DataFrame, frame_group)
        sync_index = int(sync_index_key)  # type: ignore[arg-type]

        # Check corner count
        if len(frame_df) < min_corners:
            continue

        # Check relative coverage (bbox area / max possible bbox area)
        x_min = float(frame_df["img_loc_x"].min())
        x_max = float(frame_df["img_loc_x"].max())
        y_min = float(frame_df["img_loc_y"].min())
        y_max = float(frame_df["img_loc_y"].max())
        bbox_area = (x_max - x_min) * (y_max - y_min)

        # Use relative coverage: how much of the max possible bbox is this frame using?
        relative_coverage = bbox_area / max_bbox if max_bbox > 0 else 0.0

        if relative_coverage >= min_coverage:
            eligible.append(sync_index)

    return sorted(eligible)  # Sorted for determinism


def _compute_frame_coverage(
    frame_df: pd.DataFrame,
    image_size: tuple[int, int],
    grid_size: int,
) -> set[tuple[int, int]]:
    """Compute set of (row, col) grid cells covered by frame's corners."""
    width, height = image_size
    cell_width = width / grid_size
    cell_height = height / grid_size

    covered: set[tuple[int, int]] = set()
    for _, row in frame_df.iterrows():
        col = min(int(row["img_loc_x"] / cell_width), grid_size - 1)
        row_idx = min(int(row["img_loc_y"] / cell_height), grid_size - 1)
        covered.add((row_idx, col))

    return covered


def _compute_pose_features(frame_df: pd.DataFrame, image_size: tuple[int, int]) -> np.ndarray:
    """Extract 5D pose feature vector for diversity comparison.

    Returns: [centroid_x, centroid_y, spread_x, spread_y, aspect_ratio]

    The aspect ratio (spread_x / spread_y) captures board orientation/tilt,
    which is critical for intrinsic calibration per Zhang (2000) - different
    orientations are needed for focal length observability.

    All position/spread values normalized by image dimensions for scale-invariance.
    """
    width, height = image_size

    centroid_x = frame_df["img_loc_x"].mean() / width
    centroid_y = frame_df["img_loc_y"].mean() / height

    # Use std for spread; handle edge cases
    spread_x = frame_df["img_loc_x"].std() / width if len(frame_df) > 1 else 0.0
    spread_y = frame_df["img_loc_y"].std() / height if len(frame_df) > 1 else 0.0

    # Aspect ratio encodes board tilt relative to camera
    # A tilted board has different x vs y spread in image space
    aspect_ratio = spread_x / spread_y if spread_y > 1e-6 else 1.0

    return np.array([centroid_x, centroid_y, spread_x, spread_y, aspect_ratio])


def _score_frame(
    candidate_coverage: set[tuple[int, int]],
    selected_coverage: set[tuple[int, int]],
    candidate_pose: np.ndarray,
    selected_poses: list[np.ndarray],
    grid_size: int,
    edge_weight: float = 0.2,
    corner_weight: float = 0.3,
    diversity_weight: float = 0.3,
) -> float:
    """Score a candidate frame for selection.

    Components:
    - base_coverage_gain: Count of new cells covered (not in selected_coverage)
    - edge_bonus: Extra weight for cells in edge rows/cols (0 and grid_size-1)
    - corner_bonus: Extra weight for the 4 corner cells
    - pose_diversity_bonus: Euclidean distance from nearest selected pose
    """
    new_cells = candidate_coverage - selected_coverage

    # Base coverage gain
    score = float(len(new_cells))

    # Edge bonus: cells in edge rows or columns
    edge_indices = {0, grid_size - 1}
    edge_cells = {c for c in new_cells if c[0] in edge_indices or c[1] in edge_indices}
    score += len(edge_cells) * edge_weight

    # Corner bonus: the 4 corner cells
    corners = {
        (0, 0),
        (0, grid_size - 1),
        (grid_size - 1, 0),
        (grid_size - 1, grid_size - 1),
    }
    corner_cells = new_cells & corners
    score += len(corner_cells) * corner_weight

    # Pose diversity bonus: distance from nearest selected pose
    if selected_poses:
        distances = [float(np.linalg.norm(candidate_pose - p)) for p in selected_poses]
        min_distance = min(distances)
        score += min_distance * diversity_weight

    return score


def _greedy_select(
    frame_data: dict[int, tuple[set[tuple[int, int]], np.ndarray]],
    target_count: int,
    grid_size: int,
    min_score: float = 0.01,
) -> list[int]:
    """Greedily select frames to maximize coverage score."""
    selected: list[int] = []
    selected_coverage: set[tuple[int, int]] = set()
    selected_poses: list[np.ndarray] = []

    remaining = set(frame_data.keys())

    while len(selected) < target_count and remaining:
        # Score all remaining candidates
        scores: list[tuple[float, int]] = []
        for sync_index in remaining:
            coverage, pose = frame_data[sync_index]
            score = _score_frame(coverage, selected_coverage, pose, selected_poses, grid_size)
            scores.append((score, sync_index))

        # Sort by score descending, then sync_index ascending for determinism
        scores.sort(key=lambda x: (-x[0], x[1]))

        best_score, best_frame = scores[0]
        if best_score < min_score:
            break  # Early stopping: no frame provides sufficient benefit

        # Update state
        coverage, pose = frame_data[best_frame]
        selected.append(best_frame)
        selected_coverage |= coverage
        selected_poses.append(pose)
        remaining.remove(best_frame)

    return selected


def _compute_quality_metrics(
    frame_data: dict[int, tuple[set[tuple[int, int]], np.ndarray]],
    selected_frames: list[int],
    grid_size: int,
) -> dict[str, float]:
    """Compute quality metrics for the selected frames."""
    if not selected_frames:
        return {
            "coverage_fraction": 0.0,
            "edge_coverage_fraction": 0.0,
            "corner_coverage_fraction": 0.0,
            "pose_diversity": 0.0,
        }

    # Aggregate coverage from selected frames
    total_coverage: set[tuple[int, int]] = set()
    poses: list[np.ndarray] = []
    for sync_index in selected_frames:
        coverage, pose = frame_data[sync_index]
        total_coverage |= coverage
        poses.append(pose)

    total_cells = grid_size * grid_size
    coverage_fraction = len(total_coverage) / total_cells

    # Edge cells: rows/cols 0 and grid_size-1
    edge_indices = {0, grid_size - 1}
    all_edge_cells = {
        (r, c) for r in range(grid_size) for c in range(grid_size) if r in edge_indices or c in edge_indices
    }
    covered_edge_cells = total_coverage & all_edge_cells
    edge_coverage_fraction = len(covered_edge_cells) / len(all_edge_cells) if all_edge_cells else 0.0

    # Corner cells: the 4 corners
    corner_cells = {
        (0, 0),
        (0, grid_size - 1),
        (grid_size - 1, 0),
        (grid_size - 1, grid_size - 1),
    }
    covered_corners = total_coverage & corner_cells
    corner_coverage_fraction = len(covered_corners) / len(corner_cells)

    # Pose diversity: average variance across pose dimensions
    if len(poses) > 1:
        pose_array = np.array(poses)
        pose_diversity = float(np.mean(np.var(pose_array, axis=0)))
    else:
        pose_diversity = 0.0

    return {
        "coverage_fraction": coverage_fraction,
        "edge_coverage_fraction": edge_coverage_fraction,
        "corner_coverage_fraction": corner_coverage_fraction,
        "pose_diversity": pose_diversity,
    }
