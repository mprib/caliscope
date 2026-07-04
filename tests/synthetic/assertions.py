"""
Assertion functions for comparing camera poses against ground truth.

Uses geodesic distance on SO(3) for rotation error and Euclidean distance
for translation error (comparing camera positions in world coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.alignment import estimate_similarity_transform, apply_similarity_transform
from caliscope.core.capture_volume import CaptureVolume
from caliscope.synthetic.synthetic_scene import SyntheticScene


@dataclass
class PoseError:
    """
    Error metrics between an estimated and ground truth camera pose.

    Attributes:
        rotation_deg: Geodesic distance on SO(3) in degrees.
        translation_m: Euclidean distance between camera positions in meters.
    """

    rotation_deg: float
    translation_m: float


def pose_error(estimated: CameraData, ground_truth: CameraData) -> PoseError:
    """
    Compute rotation and translation errors between camera poses.

    Rotation error uses geodesic distance on SO(3):
    - Compute relative rotation R_rel = R_est @ R_gt.T
    - Extract angle from R_rel using Rodrigues representation

    Translation error is Euclidean distance between camera positions:
    - Camera position in world = -R.T @ t
    """
    if estimated.rotation is None or estimated.translation is None:
        raise ValueError(f"Estimated camera {estimated.cam_id} lacks pose")
    if ground_truth.rotation is None or ground_truth.translation is None:
        raise ValueError(f"Ground truth camera {ground_truth.cam_id} lacks pose")

    # Rotation error: geodesic distance on SO(3)
    R_rel: np.ndarray = estimated.rotation @ ground_truth.rotation.T
    rodrigues, _ = cv2.Rodrigues(R_rel)
    rotation_rad = np.linalg.norm(rodrigues)
    rotation_deg = float(np.degrees(rotation_rad))

    # Translation error: Euclidean distance between camera positions
    pos_est = -estimated.rotation.T @ estimated.translation
    pos_gt = -ground_truth.rotation.T @ ground_truth.translation
    translation_m = float(np.linalg.norm(pos_est - pos_gt))

    return PoseError(rotation_deg=rotation_deg, translation_m=translation_m)


def cameras_match_ground_truth(
    actual: CameraArray,
    expected: CameraArray,
    rotation_tol_deg: float = 0.5,
    translation_tol_m: float = 0.005,
    skip_ports: list[int] | None = None,
) -> tuple[bool, str]:
    """
    Check if optimized cameras match ground truth within tolerances.

    Returns:
        (success, message) tuple where message contains failure details if any.
    """
    if skip_ports is None:
        skip_ports = []

    failures = []

    for cam_id in expected.cameras:
        if cam_id in skip_ports:
            continue

        if cam_id not in actual.cameras:
            failures.append(f"Camera {cam_id}: missing from actual")
            continue

        error = pose_error(actual.cameras[cam_id], expected.cameras[cam_id])

        if error.rotation_deg > rotation_tol_deg:
            failures.append(
                f"Camera {cam_id}: rotation error {error.rotation_deg:.3f} deg > {rotation_tol_deg} deg tolerance"
            )

        if error.translation_m > translation_tol_m:
            failures.append(
                f"Camera {cam_id}: translation error {error.translation_m:.4f} m > {translation_tol_m} m tolerance"
            )

    if failures:
        return False, "\n".join(failures)
    return True, ""


def _camera_centers(camera_array: CameraArray) -> dict[int, np.ndarray]:
    """Extract world-space camera centers: C = -R^T @ t."""
    centers = {}
    for cam_id, cam in camera_array.cameras.items():
        if cam.rotation is not None and cam.translation is not None:
            centers[cam_id] = -cam.rotation.T @ cam.translation
    return centers


def _points_are_collinear(pts: np.ndarray, tol: float = 1e-3) -> bool:
    """Check if an Nx3 point set is approximately collinear."""
    centered = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return bool(s[1] / s[0] < tol) if s[0] > 0 else True


def align_to_ground_truth(optimized: CaptureVolume, scene: SyntheticScene) -> CaptureVolume:
    """Similarity-align an optimized volume to the scene's ground truth.

    With >= 3 non-collinear posed cameras: Umeyama on camera centers.
    Camera centers are few but well-spread in 3D for ring geometries,
    unlike world points which are often near-coplanar (planar board,
    thin trajectory slab).

    Falls back to world-point alignment when camera centers are
    insufficient: fewer than 3, or near-collinear (e.g. a camera line).
    Collinear cameras define a line, not a plane, so they cannot fix
    the rotation about the line axis.
    """
    opt_centers = _camera_centers(optimized.camera_array)
    gt_centers = _camera_centers(scene.camera_array)
    shared_ids = sorted(set(opt_centers) & set(gt_centers))

    use_cameras = len(shared_ids) >= 3
    if use_cameras:
        gt_pts_check = np.array([gt_centers[cid] for cid in shared_ids])
        if _points_are_collinear(gt_pts_check):
            use_cameras = False

    if use_cameras:
        opt_pts = np.array([opt_centers[cid] for cid in shared_ids])
        gt_pts = np.array([gt_centers[cid] for cid in shared_ids])
        sim = estimate_similarity_transform(opt_pts, gt_pts)
    else:
        gt_df = scene.world_points.df
        opt_df = optimized.world_points.df
        merged = gt_df.merge(
            opt_df,
            on=["sync_index", "object_id", "keypoint_id"],
            suffixes=("_gt", "_opt"),
            how="inner",
        )
        gt_pts = merged[["x_coord_gt", "y_coord_gt", "z_coord_gt"]].to_numpy()
        opt_pts = merged[["x_coord_opt", "y_coord_opt", "z_coord_opt"]].to_numpy()
        sim = estimate_similarity_transform(opt_pts, gt_pts)

    cameras, world_points = apply_similarity_transform(optimized.camera_array, optimized.world_points, sim)
    return CaptureVolume(cameras, optimized.image_points, world_points, optimized.constraints)


def assert_cameras_moved(
    initial: CameraArray,
    final: CameraArray,
    skip_ports: list[int] | None = None,
    min_movement: float = 1e-6,
) -> None:
    """
    Assert that cameras moved during optimization.

    This catches the critical bug where camera parameters are unpacked
    but never assigned back.
    """
    if skip_ports is None:
        skip_ports = []

    for cam_id in initial.cameras:
        if cam_id in skip_ports:
            continue

        initial_vec = initial.cameras[cam_id].extrinsics_to_vector()
        final_vec = final.cameras[cam_id].extrinsics_to_vector()

        movement = np.linalg.norm(final_vec - initial_vec)

        assert movement > min_movement, (
            f"Camera {cam_id} didn't move during optimization! Parameter change: {movement:.2e}"
        )
