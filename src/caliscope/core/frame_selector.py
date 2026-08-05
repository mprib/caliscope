"""Automatic frame selection for intrinsic camera calibration.

Implements a two-phase deterministic algorithm for selecting calibration frames:

Phase 1 (Orientation Diversity): Ensures minimum orientation diversity by
selecting anchor frames from distinct tilt directions. This is a HARD CONSTRAINT
for focal length observability - without sufficient orientation diversity, the
focal length and principal point parameters become coupled (Zhang 2000).

Phase 2 (Spatial Coverage): Fills remaining slots with greedy coverage
optimization, prioritizing edge/corner regions (critical for distortion estimation).

Literature foundation:
- Zhang (2000): Different orientations needed for focal length observability
- MATLAB/OpenCV docs: Peripheral coverage constrains distortion model
- Krause (2012): Greedy submodular maximization achieves (1-1/e) approximation

Orientation features are extracted from the homography mapping board coordinates
to image coordinates. The perspective components of H encode board tilt relative
to the image plane.
"""

from dataclasses import dataclass
import logging
from typing import NamedTuple, cast

import cv2
import numpy as np
import pandas as pd

from caliscope.core.point_data import ImagePoints

logger = logging.getLogger(__name__)

# --- Type Aliases for Domain Clarity ---
GridCell = tuple[int, int]  # (row, col) in coverage grid
CoveredCells = set[GridCell]
PoseFeatures = np.ndarray  # Shape: (5,) - see indices below
ObjectBounds = tuple[np.ndarray, np.ndarray]  # (minimum XY, XY range)


class OrientationFeatures(NamedTuple):
    """Board orientation extracted from homography.

    These features capture how the calibration board is oriented relative to
    the camera, which is critical for focal length observability (Zhang 2000).
    """

    tilt_direction: float  # Angle in radians [0, 2π) - direction board is tilting
    tilt_magnitude: float  # Scalar [0, inf) - how tilted the board is (0 = frontal)
    in_plane_rotation: float  # Angle in radians [0, 2π) - rotation around optical axis


# Pose feature indices (for documentation and potential direct access)
_POSE_CENTROID_X = 0
_POSE_CENTROID_Y = 1
_POSE_SPREAD_X = 2
_POSE_SPREAD_Y = 3
_POSE_ASPECT_RATIO = 4

# Orientation binning constants
_NUM_TILT_DIRECTION_BINS = 8  # 45° sectors
MIN_TILT_FOR_DIVERSITY = 0.02  # About a 2% projective scale gradient across the full board
MIN_CORNERS_FOR_ORIENTATION = 12
_HOMOGRAPHY_RANSAC_THRESHOLD_PX = 3.0
_MIN_GEOMETRY_INLIER_FRACTION = 0.85
_MAX_NORMALIZED_PLANAR_RMS = 0.08


@dataclass(frozen=True)
class FrameGeometryQuality:
    """Robust planar-correspondence diagnostics for one calibration frame."""

    inlier_positions: tuple[int, ...]
    inlier_fraction: float
    planar_rms_px: float
    normalized_planar_rms: float
    score: float


@dataclass(frozen=True)
class FrameCoverageData:
    """Precomputed coverage, pose, and orientation data for a single eligible frame."""

    covered_cells: CoveredCells
    pose_features: PoseFeatures
    orientation: OrientationFeatures
    quality_score: float


@dataclass(frozen=True)
class IntrinsicCoverageReport:
    """Coverage and selection result for intrinsic calibration.

    Assesses whether a single camera has sufficient data for intrinsic calibration:
    - Spatial grid coverage (5x5 grid)
    - Board orientation diversity (Zhang 2000)
    """

    selected_frames: list[int]  # sync_index values

    # Quality metrics (for diagnostics/logging)
    coverage_fraction: float  # Target > 0.80 (20+ of 25 cells)
    edge_coverage_fraction: float  # Target > 0.75
    corner_coverage_fraction: float  # Target > 0.50
    pose_diversity: float  # Variance of pose features

    # Orientation diversity (critical for focal length observability)
    orientation_sufficient: bool  # True if ≥4 distinct tilt directions found
    orientation_count: int  # Number of distinct orientation bins covered

    # Selection metadata
    eligible_frame_count: int
    total_frame_count: int


def select_calibration_frames(
    image_points: ImagePoints,
    cam_id: int,
    image_size: tuple[int, int],
    *,
    target_frame_count: int = 30,
    min_corners_per_frame: int = 6,
    min_orientations: int = 4,
    grid_size: int = 5,
) -> IntrinsicCoverageReport:
    """Select optimal frames for intrinsic camera calibration.

    Uses a two-phase selection algorithm:

    Phase 1 (Orientation Diversity): Ensures minimum orientation diversity
    by selecting anchor frames from distinct tilt directions. This is a
    HARD CONSTRAINT - without sufficient orientation diversity, focal length
    estimation is mathematically ill-posed (Zhang 2000).

    Phase 2 (Spatial Coverage): Fills remaining slots by greedily optimizing
    for image grid coverage, with extra weight for edge/corner regions
    (critical for distortion estimation).

    Determinism: Ties are broken by sync_index (lowest first). No random sampling.

    Args:
        image_points: Detected charuco corners across all frames
        cam_id: Camera cam_id to select frames for
        image_size: (width, height) of the camera image
        target_frame_count: Maximum frames to select (default 30)
        min_corners_per_frame: Minimum corners required per frame (default 6)
        min_orientations: Minimum distinct tilt directions required (default 4)
        grid_size: Coverage grid dimension (default 5 for 5x5 = 25 cells)

    Returns:
        IntrinsicCoverageReport with selected sync_index values, quality metrics,
        and orientation_sufficient flag indicating if diversity requirements were met
    """
    # Filter to specified cam_id
    cam_df = cast(pd.DataFrame, image_points.df[image_points.df["cam_id"] == cam_id].copy())
    total_frame_count = int(cam_df["sync_index"].nunique())

    if total_frame_count == 0:
        return IntrinsicCoverageReport(
            selected_frames=[],
            coverage_fraction=0.0,
            edge_coverage_fraction=0.0,
            corner_coverage_fraction=0.0,
            pose_diversity=0.0,
            orientation_sufficient=False,
            orientation_count=0,
            eligible_frame_count=0,
            total_frame_count=0,
        )

    # Reject point sets that cannot represent a coherent view of one plane.
    eligible_frames = _filter_eligible_frames(cam_df, min_corners_per_frame)

    if not eligible_frames:
        return IntrinsicCoverageReport(
            selected_frames=[],
            coverage_fraction=0.0,
            edge_coverage_fraction=0.0,
            corner_coverage_fraction=0.0,
            pose_diversity=0.0,
            orientation_sufficient=False,
            orientation_count=0,
            eligible_frame_count=0,
            total_frame_count=total_frame_count,
        )

    # Precompute coverage, pose, and orientation features for all eligible frames
    frame_data: dict[int, FrameCoverageData] = {}
    all_obj_points = cam_df[["obj_loc_x", "obj_loc_y"]].to_numpy(dtype=np.float32)
    object_minimum = all_obj_points.min(axis=0)
    object_range = all_obj_points.max(axis=0) - object_minimum
    object_range[object_range < 1e-6] = 1.0
    object_bounds = (object_minimum, object_range)
    maximum_inlier_count = max(len(geometry.inlier_positions) for geometry in eligible_frames.values())
    for sync_index, geometry in eligible_frames.items():
        frame_df = cast(pd.DataFrame, cam_df[cam_df["sync_index"] == sync_index])
        inlier_df = cast(pd.DataFrame, frame_df.iloc[list(geometry.inlier_positions)])
        coverage = _compute_frame_coverage(inlier_df, image_size, grid_size)
        pose = _compute_pose_features(inlier_df, image_size)
        orientation = (
            _compute_orientation_features(inlier_df, object_bounds=object_bounds)
            if len(inlier_df) >= MIN_CORNERS_FOR_ORIENTATION
            else OrientationFeatures(0.0, 0.0, 0.0)
        )
        corner_fraction = len(inlier_df) / maximum_inlier_count
        corner_quality = 0.1 + 0.9 * float(np.sqrt(corner_fraction))
        frame_data[sync_index] = FrameCoverageData(
            coverage,
            pose,
            orientation,
            geometry.score * corner_quality,
        )

    logger.info(
        "Intrinsic geometry gate for cam_id %s: accepted %d/%d detected frames",
        cam_id,
        len(eligible_frames),
        total_frame_count,
    )

    # Two-phase selection
    # Phase 1: Select orientation anchor frames (hard constraint for focal length observability)
    anchor_frames, covered_bins = _select_orientation_anchors(frame_data, min_orientations)
    orientation_count = len(covered_bins)
    orientation_sufficient = orientation_count >= min_orientations

    # Phase 2: Fill remaining slots with coverage-optimized selection
    remaining_budget = target_frame_count - len(anchor_frames)
    if remaining_budget > 0:
        coverage_frames = _greedy_select_coverage(
            frame_data,
            already_selected=anchor_frames,
            target_count=remaining_budget,
            grid_size=grid_size,
        )
        selected_frames = anchor_frames + coverage_frames
    else:
        selected_frames = anchor_frames[:target_frame_count]

    # Compute quality metrics
    metrics = _compute_quality_metrics(frame_data, selected_frames, grid_size)

    return IntrinsicCoverageReport(
        selected_frames=selected_frames,
        coverage_fraction=metrics["coverage_fraction"],
        edge_coverage_fraction=metrics["edge_coverage_fraction"],
        corner_coverage_fraction=metrics["corner_coverage_fraction"],
        pose_diversity=metrics["pose_diversity"],
        orientation_sufficient=orientation_sufficient,
        orientation_count=orientation_count,
        eligible_frame_count=len(eligible_frames),
        total_frame_count=total_frame_count,
    )


def _filter_eligible_frames(
    cam_df: pd.DataFrame,
    min_corners: int,
) -> dict[int, FrameGeometryQuality]:
    """Return frames with enough geometrically coherent planar correspondences.

    Corner count alone admits false marker identities and subpixel-collapse
    failures.  A RANSAC homography verifies that most observations can arise
    from one planar board.  RMS is normalized by local point pitch so this gate
    works across video resolutions without rejecting small, distant boards.
    """
    eligible: dict[int, FrameGeometryQuality] = {}
    grouped = cam_df.groupby("sync_index")
    for sync_index_key, frame_group in grouped:
        frame_df = cast(pd.DataFrame, frame_group)
        sync_index = int(sync_index_key)  # type: ignore[arg-type]
        quality = _assess_frame_geometry(frame_df, min_corners)
        if quality is not None:
            eligible[sync_index] = quality

    return dict(sorted(eligible.items()))


def _assess_frame_geometry(
    frame_df: pd.DataFrame,
    min_corners: int,
) -> FrameGeometryQuality | None:
    """Assess whether a frame's IDs and pixels form a credible planar grid."""
    if len(frame_df) < min_corners:
        return None

    identity_columns = [column for column in ("object_id", "keypoint_id") if column in frame_df.columns]
    if identity_columns and frame_df.duplicated(subset=identity_columns).any():
        return None

    obj_points = frame_df[["obj_loc_x", "obj_loc_y"]].to_numpy(dtype=np.float32)
    img_points = frame_df[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32)
    if not np.isfinite(obj_points).all() or not np.isfinite(img_points).all():
        return None

    obj_range = obj_points.max(axis=0) - obj_points.min(axis=0)
    if np.any(obj_range < 1e-6):
        return None
    obj_normalized = (obj_points - obj_points.min(axis=0)) / obj_range

    homography, mask = cv2.findHomography(
        obj_normalized,
        img_points,
        cv2.RANSAC,
        _HOMOGRAPHY_RANSAC_THRESHOLD_PX,
    )
    if homography is None or mask is None:
        return None

    inlier_mask = mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_fraction = inlier_count / len(frame_df)
    if inlier_count < min_corners or inlier_fraction < _MIN_GEOMETRY_INLIER_FRACTION:
        return None

    # Refit without RANSAC on verified correspondences before measuring RMS.
    homography, _ = cv2.findHomography(obj_normalized[inlier_mask], img_points[inlier_mask], 0)
    if homography is None:
        return None
    projected = cv2.perspectiveTransform(obj_normalized[inlier_mask, None, :], homography)[:, 0, :]
    residuals = np.linalg.norm(projected - img_points[inlier_mask], axis=1)
    planar_rms_px = float(np.sqrt(np.mean(residuals**2)))

    # Nearest-neighbor pitch is topology-agnostic and remains useful for
    # ChArUco partial detections and complete chessboards alike.
    inlier_img = img_points[inlier_mask]
    pairwise = np.linalg.norm(inlier_img[:, None, :] - inlier_img[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    local_pitch_px = float(np.median(pairwise.min(axis=1)))
    if not np.isfinite(local_pitch_px) or local_pitch_px < 1.0:
        return None

    normalized_rms = planar_rms_px / local_pitch_px
    if not np.isfinite(normalized_rms) or normalized_rms > _MAX_NORMALIZED_PLANAR_RMS:
        return None

    # Keep the score in (0, 1]; it breaks selection ties in favor of coherent
    # frames while spatial and orientation diversity remain the primary goals.
    quality_score = inlier_fraction / (1.0 + normalized_rms)
    return FrameGeometryQuality(
        inlier_positions=tuple(int(position) for position in np.flatnonzero(inlier_mask)),
        inlier_fraction=inlier_fraction,
        planar_rms_px=planar_rms_px,
        normalized_planar_rms=normalized_rms,
        score=quality_score,
    )


def _compute_frame_coverage(
    frame_df: pd.DataFrame,
    image_size: tuple[int, int],
    grid_size: int,
) -> CoveredCells:
    """Compute set of (row, col) grid cells covered by frame's corners."""
    width, height = image_size
    cell_width = width / grid_size
    cell_height = height / grid_size

    covered: CoveredCells = set()
    for _, point in frame_df.iterrows():
        grid_col = max(0, min(int(point["img_loc_x"] / cell_width), grid_size - 1))
        grid_row = max(0, min(int(point["img_loc_y"] / cell_height), grid_size - 1))
        covered.add((grid_row, grid_col))

    return covered


def _compute_pose_features(frame_df: pd.DataFrame, image_size: tuple[int, int]) -> PoseFeatures:
    """Extract 5D pose feature vector for diversity comparison.

    Returns: [centroid_x, centroid_y, spread_x, spread_y, aspect_ratio]
    (See _POSE_* constants for index mapping)

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


def _compute_orientation_features(
    frame_df: pd.DataFrame,
    *,
    object_bounds: ObjectBounds | None = None,
) -> OrientationFeatures:
    """Extract board orientation from 2D-2D homography.

    Computes a homography mapping board object coordinates to image coordinates,
    then extracts orientation features from the homography matrix structure:

    - tilt_direction: Direction the board is tilting (from perspective components)
    - tilt_magnitude: How much the board is tilted (0 = frontal-parallel)
    - in_plane_rotation: Rotation around the camera's optical axis

    These features capture what Zhang (2000) showed is critical for focal length
    observability: boards at different orientations provide constraints that
    "pull apart" the focal length and principal point parameter coupling.

    The homography H relates board coords (obj_loc) to image coords (img_loc):
        [u, v, 1]^T ~ H @ [X, Y, 1]^T

    H[2,0] and H[2,1] encode perspective distortion (tilt information).
    The affine component (top-left 2x2) encodes scale, rotation, and shear.
    """
    # Extract object (board) and image coordinates
    obj_points = frame_df[["obj_loc_x", "obj_loc_y"]].to_numpy(dtype=np.float32)
    img_points = frame_df[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32)

    # Need at least 4 points for homography
    if len(obj_points) < 4:
        return OrientationFeatures(
            tilt_direction=0.0,
            tilt_magnitude=0.0,
            in_plane_rotation=0.0,
        )

    # Normalize object coordinates to [0, 1] so tilt_magnitude reflects pure
    # geometric foreshortening (near edge vs far edge size ratio), independent
    # of whatever coordinate units the board uses.
    if object_bounds is None:
        obj_minimum = obj_points.min(axis=0)
        obj_range = obj_points.max(axis=0) - obj_minimum
        obj_range[obj_range < 1e-6] = 1.0  # guard against degenerate axis
    else:
        obj_minimum, obj_range = object_bounds
    obj_normalized = (obj_points - obj_minimum) / obj_range

    # Compute homography using RANSAC for robustness
    H, mask = cv2.findHomography(
        obj_normalized,
        img_points,
        cv2.RANSAC,
        _HOMOGRAPHY_RANSAC_THRESHOLD_PX,
    )

    if H is None or mask is None:
        return OrientationFeatures(
            tilt_direction=0.0,
            tilt_magnitude=0.0,
            in_plane_rotation=0.0,
        )

    inlier_mask = mask.reshape(-1).astype(bool)
    if int(inlier_mask.sum()) < MIN_CORNERS_FOR_ORIENTATION:
        return OrientationFeatures(0.0, 0.0, 0.0)
    H, _ = cv2.findHomography(obj_normalized[inlier_mask], img_points[inlier_mask], 0)
    if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return OrientationFeatures(0.0, 0.0, 0.0)

    # Normalize H by H[2,2] to get standard form
    H = H / H[2, 2]

    # Extract tilt from perspective components
    # H[2,0] and H[2,1] encode how the board plane tilts relative to the image plane
    tilt_direction = float(np.arctan2(H[2, 1], H[2, 0]))
    if tilt_direction < 0:
        tilt_direction += 2 * np.pi

    tilt_magnitude = float(np.sqrt(H[2, 0] ** 2 + H[2, 1] ** 2))

    # Extract in-plane rotation from affine component
    # The top-left 2x2 of H approximates an affine transformation
    A = H[:2, :2]
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt  # Rotation component

    in_plane_rotation = float(np.arctan2(R[1, 0], R[0, 0]))
    if in_plane_rotation < 0:
        in_plane_rotation += 2 * np.pi

    return OrientationFeatures(
        tilt_direction=tilt_direction,
        tilt_magnitude=tilt_magnitude,
        in_plane_rotation=in_plane_rotation,
    )


def _get_orientation_bin(orientation: OrientationFeatures) -> int | None:
    """Map orientation to a bin index for diversity checking.

    Returns None for frontal-parallel boards (tilt below threshold).
    Returns bin index 0-7 for tilted boards (8 sectors of 45° each).
    """
    if orientation.tilt_magnitude < MIN_TILT_FOR_DIVERSITY:
        return None  # Frontal-parallel, doesn't contribute to orientation diversity

    # Map tilt_direction [0, 2π) to bin [0, 7]
    bin_index = int(orientation.tilt_direction / (2 * np.pi) * _NUM_TILT_DIRECTION_BINS)
    return min(bin_index, _NUM_TILT_DIRECTION_BINS - 1)  # Clamp to valid range


def _score_frame(
    candidate_coverage: CoveredCells,
    selected_coverage: CoveredCells,
    candidate_pose: PoseFeatures,
    selected_poses: list[PoseFeatures],
    grid_size: int,
    quality_score: float = 1.0,
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

    return score * (0.5 + 0.5 * quality_score) + 0.01 * quality_score


def _select_orientation_anchors(
    frame_data: dict[int, FrameCoverageData],
    min_orientations: int,
) -> tuple[list[int], set[int]]:
    """Phase 1: Select anchor frames ensuring orientation diversity.

    Selects frames from distinct tilt direction bins to ensure the calibration
    problem is well-posed for focal length estimation. This is a HARD CONSTRAINT
    per Zhang (2000) - without sufficient orientation diversity, the focal length
    and principal point parameters become coupled.

    Strategy:
    - Bin frames by tilt direction (8 sectors of 45° each)
    - For each occupied bin, select the frame with highest tilt magnitude
      (more tilted = more information for focal length estimation)
    - Frontal-parallel frames (low tilt) don't count toward diversity

    Returns:
        selected_anchors: List of sync_index values for anchor frames
        covered_bins: Set of bin indices that have been covered
    """
    # Group frames by orientation bin
    bin_to_frames: dict[int, list[tuple[int, float, float]]] = {}

    for sync_index, data in frame_data.items():
        bin_idx = _get_orientation_bin(data.orientation)
        if bin_idx is not None:  # Skip frontal-parallel frames
            if bin_idx not in bin_to_frames:
                bin_to_frames[bin_idx] = []
            bin_to_frames[bin_idx].append(
                (sync_index, data.orientation.tilt_magnitude, data.quality_score)
            )

    # Select one frame per bin, preferring higher tilt magnitude
    selected_anchors: list[int] = []
    covered_bins: set[int] = set()

    # Sort bins by index for determinism
    for bin_idx in sorted(bin_to_frames.keys()):
        frames = bin_to_frames[bin_idx]
        # Prefer informative tilt only among geometrically reliable frames.
        frames.sort(key=lambda x: (-(x[1] * x[2]), -x[2], x[0]))
        best_frame = frames[0][0]
        selected_anchors.append(best_frame)
        covered_bins.add(bin_idx)

    return selected_anchors, covered_bins


def _greedy_select_coverage(
    frame_data: dict[int, FrameCoverageData],
    already_selected: list[int],
    target_count: int,
    grid_size: int,
    min_score: float = 0.01,
) -> list[int]:
    """Phase 2: Greedily select frames to maximize spatial coverage.

    After orientation anchors are selected, this fills remaining slots by
    optimizing for spatial coverage with edge/corner bonuses.
    """
    # Initialize with already-selected anchor frames
    selected_coverage: CoveredCells = set()
    selected_poses: list[PoseFeatures] = []

    for sync_index in already_selected:
        data = frame_data[sync_index]
        selected_coverage |= data.covered_cells
        selected_poses.append(data.pose_features)

    # Exclude already-selected frames
    remaining = set(frame_data.keys()) - set(already_selected)
    newly_selected: list[int] = []

    while len(newly_selected) < target_count and remaining:
        # Score all remaining candidates
        scores: list[tuple[float, int]] = []
        for sync_index in remaining:
            data = frame_data[sync_index]
            score = _score_frame(
                data.covered_cells,
                selected_coverage,
                data.pose_features,
                selected_poses,
                grid_size,
                quality_score=data.quality_score,
            )
            scores.append((score, sync_index))

        # Sort by score descending, then sync_index ascending for determinism
        scores.sort(key=lambda x: (-x[0], x[1]))

        best_score, best_frame = scores[0]
        if best_score < min_score:
            break  # Early stopping: no frame provides sufficient benefit

        # Update state
        best_data = frame_data[best_frame]
        newly_selected.append(best_frame)
        selected_coverage |= best_data.covered_cells
        selected_poses.append(best_data.pose_features)
        remaining.remove(best_frame)

    return newly_selected


def _compute_quality_metrics(
    frame_data: dict[int, FrameCoverageData],
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
    total_coverage: CoveredCells = set()
    poses: list[PoseFeatures] = []
    for sync_index in selected_frames:
        data = frame_data[sync_index]
        total_coverage |= data.covered_cells
        poses.append(data.pose_features)

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
