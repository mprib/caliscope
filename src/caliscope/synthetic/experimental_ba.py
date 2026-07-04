# TEMPORARY: delete when production pixel-space BA lands (roadmap Phase 3).
# See tasks.json: delete-experimental-ba.
"""Experimental bundle adjustment variants for testbed use.

Zero production code changes. These functions replicate the production
optimize() setup and vary only the solver parameters (loss function,
free intrinsic set). When the roadmap's production implementation lands,
the experiments migrate onto it and this module is deleted.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from caliscope.core.capture_volume import CaptureVolume, OptimizationStatus
from caliscope.core.point_data import WorldPoints
from caliscope.core.reprojection import (
    CameraIndices,
    ImageCoords,
    bundle_residuals,
)

logger = logging.getLogger(__name__)

N_CAM_PARAMS = 8  # rvec(3) + tvec(3) + f(1) + k1(1)

_SCIPY_STATUS_REASONS = {
    -1: "improper_input",
    0: "max_evaluations",
    1: "converged_gtol",
    2: "converged_ftol",
    3: "converged_xtol",
    4: "converged_ftol_and_xtol",
}


@dataclass(frozen=True)
class IntrinsicEstimate:
    cam_id: int
    f_recovered: float
    k1_recovered: float
    f_initial: float
    k1_initial: float


@dataclass(frozen=True)
class JointOptimizationResult:
    capture_volume: CaptureVolume
    intrinsic_estimates: tuple[IntrinsicEstimate, ...]
    converged: bool
    hit_bounds: bool
    final_cost: float


def optimize_with_robust_loss(
    capture_volume: CaptureVolume,
    *,
    loss: str = "huber",
    f_scale_px: float = 1.0,
    ftol: float = 1e-8,
    max_nfev: int = 1000,
    verbose: int = 0,
    strict: bool = True,
    use_constraints: bool = True,
    pixel_sigma: float = 1.0,
) -> CaptureVolume:
    """Production-layout BA with a robust loss function.

    Replicates CaptureVolume.optimize() exactly — same residuals, sparsity,
    parameter layout — and changes only the scipy loss and f_scale.

    f_scale_px is the inlier scale in pixels. Production residuals are in
    normalized coordinates, so f_scale_px is divided by the median focal
    length to convert: 1.0 px ≈ 7.2e-4 normalized at the WEBCAM profile.
    """
    matched_mask = capture_volume.img_to_obj_map >= 0
    posed_cam_ids = set(capture_volume.camera_array.posed_cam_id_to_index.keys())
    posed_mask: np.ndarray = capture_volume.image_points.df["cam_id"].isin(posed_cam_ids).to_numpy()
    combined_mask = matched_mask & posed_mask

    matched_img_df = capture_volume.image_points.df[combined_mask]

    camera_indices: CameraIndices = np.array(
        [capture_volume.camera_array.posed_cam_id_to_index[cam_id] for cam_id in matched_img_df["cam_id"]],
        dtype=np.int16,
    )
    image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
    image_to_world_indices = capture_volume.img_to_obj_map[combined_mask]

    initial_params = capture_volume._get_vectorized_params()

    constraint_pairs = None
    constraint_distances = None
    constraint_weights = None

    focal_lengths = [
        cam.matrix[0, 0] for cam in capture_volume.camera_array.posed_cameras.values() if cam.matrix is not None
    ]
    f_median = float(np.median(focal_lengths))

    if use_constraints and capture_volume.constraints is not None:
        arrays = capture_volume._build_constraint_arrays()
        if arrays is not None:
            constraint_pairs, constraint_distances, constraint_sigmas = arrays
            constraint_weights = (pixel_sigma / f_median) / constraint_sigmas

    sparsity_pattern = capture_volume._get_sparsity_pattern(camera_indices, image_to_world_indices, constraint_pairs)

    # Convert pixel-space f_scale to normalized-coordinate units
    f_scale_normalized = f_scale_px / f_median

    result = least_squares(
        bundle_residuals,
        initial_params,
        args=(
            capture_volume.camera_array,
            camera_indices,
            image_coords,
            image_to_world_indices,
            True,
            constraint_pairs,
            constraint_distances,
            constraint_weights,
        ),
        jac_sparsity=sparsity_pattern,
        verbose=verbose,
        x_scale="jac",
        loss=loss,
        f_scale=f_scale_normalized,
        ftol=ftol,
        max_nfev=max_nfev,
        method="trf",
    )

    termination_reason = _SCIPY_STATUS_REASONS.get(result.status, f"unknown_{result.status}")
    converged = result.status in (1, 2, 3, 4)

    optimization_status = OptimizationStatus(
        converged=converged,
        termination_reason=termination_reason,
        iterations=result.nfev,
        final_cost=float(result.cost),
    )

    if strict and not converged:
        from caliscope.exceptions import CalibrationError

        raise CalibrationError(
            f"Robust BA did not converge: {termination_reason}\n"
            f"Pass strict=False to suppress this error and inspect the result."
        )

    new_camera_array = deepcopy(capture_volume.camera_array)
    new_camera_array.update_extrinsic_params(result.x)

    n_cams = len(capture_volume.camera_array.posed_cameras)
    n_cam_params = 6
    optimized_points = result.x[n_cams * n_cam_params :].reshape((-1, 3))

    new_world_df = capture_volume.world_points.df.copy()
    new_world_df[["x_coord", "y_coord", "z_coord"]] = optimized_points

    return CaptureVolume(
        camera_array=new_camera_array,
        image_points=capture_volume.image_points,
        world_points=WorldPoints(new_world_df),
        constraints=capture_volume.constraints,
        _optimization_status=optimization_status,
    )


def _joint_residuals(
    params: NDArray[np.float64],
    n_cams: int,
    camera_indices: CameraIndices,
    image_coords: ImageCoords,
    obj_indices: NDArray[np.int32],
    cx_cy: NDArray[np.float64],
    dist_tail: NDArray[np.float64],
    f_initial: NDArray[np.float64],
    constraint_pairs: NDArray[np.int32] | None = None,
    constraint_distances: NDArray[np.float64] | None = None,
    constraint_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Pixel-space reprojection residuals with free per-camera f and k1.

    Unlike bundle_residuals (normalized coordinates, fixed intrinsics), this
    projects with cv2.projectPoints using each camera's trial f/k1 and its
    fixed cx, cy, and remaining distortion terms. Residuals are scaled by
    1/f_initial (the camera's starting focal length, not the trial value) so
    the weighting is stable across solver iterations and comparable across
    cameras of differing focal length.
    """
    camera_params = params[: n_cams * N_CAM_PARAMS].reshape((n_cams, N_CAM_PARAMS))
    points_3d = params[n_cams * N_CAM_PARAMS :].reshape((-1, 3))
    world_coords = points_3d[obj_indices]

    errors_xy = np.zeros_like(image_coords)

    for cam_idx in range(n_cams):
        cam_mask = camera_indices == cam_idx
        if not cam_mask.any():
            continue

        rvec = camera_params[cam_idx, 0:3]
        tvec = camera_params[cam_idx, 3:6]
        f = camera_params[cam_idx, 6]
        k1 = camera_params[cam_idx, 7]
        cx, cy = cx_cy[cam_idx]

        cam_matrix = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([k1, *dist_tail[cam_idx]])

        cam_world_coords = world_coords[cam_mask].reshape(-1, 1, 3)
        cam_observed = image_coords[cam_mask]

        projected, _ = cv2.projectPoints(cam_world_coords, rvec, tvec, cam_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)

        errors_xy[cam_mask] = (projected - cam_observed) / f_initial[cam_idx]

    reproj = errors_xy.ravel()

    if constraint_pairs is not None:
        diffs = points_3d[constraint_pairs[:, 0]] - points_3d[constraint_pairs[:, 1]]
        constraint_residuals = (np.linalg.norm(diffs, axis=1) - constraint_distances) * constraint_weights
        return np.concatenate([reproj, constraint_residuals])

    return reproj


def _joint_sparsity_pattern(
    camera_indices: NDArray[np.int16],
    obj_indices: NDArray[np.int32],
    n_cameras: int,
    n_points: int,
    constraint_pairs: NDArray[np.int32] | None = None,
) -> lil_matrix:
    """Sparsity for the 8-param-per-camera joint BA Jacobian.

    Reprojection rows depend on all 8 camera params (rvec, tvec, f, k1) plus
    3 point params. Constraint rows depend only on 6 point params.
    """
    n_observations = len(camera_indices)
    n_constraints = len(constraint_pairs) if constraint_pairs is not None else 0
    n_residuals = n_observations * 2 + n_constraints
    n_params = n_cameras * N_CAM_PARAMS + n_points * 3

    sparsity = lil_matrix((n_residuals, n_params), dtype=int)
    obs_idx = np.arange(n_observations)

    for cam_param in range(N_CAM_PARAMS):
        param_col = camera_indices * N_CAM_PARAMS + cam_param
        sparsity[2 * obs_idx, param_col] = 1
        sparsity[2 * obs_idx + 1, param_col] = 1

    for point_param in range(3):
        param_col = n_cameras * N_CAM_PARAMS + obj_indices * 3 + point_param
        sparsity[2 * obs_idx, param_col] = 1
        sparsity[2 * obs_idx + 1, param_col] = 1

    if constraint_pairs is not None:
        c_idx = np.arange(n_constraints)
        row_offset = n_observations * 2
        for coord in range(3):
            col_a = n_cameras * N_CAM_PARAMS + constraint_pairs[:, 0] * 3 + coord
            col_b = n_cameras * N_CAM_PARAMS + constraint_pairs[:, 1] * 3 + coord
            sparsity[row_offset + c_idx, col_a] = 1
            sparsity[row_offset + c_idx, col_b] = 1

    return sparsity


def optimize_with_free_intrinsics(
    capture_volume: CaptureVolume,
    *,
    ftol: float = 1e-8,
    max_nfev: int = 2000,
    verbose: int = 0,
    strict: bool = True,
    use_constraints: bool = True,
    pixel_sigma: float = 1.0,
) -> JointOptimizationResult:
    """Joint BA optimizing extrinsics plus per-camera focal length and k1.

    8 params per camera: [rvec(3), tvec(3), f, k1]. Residuals are computed
    in pixel space via cv2.projectPoints (see _joint_residuals), not the
    normalized-coordinate bundle_residuals used by production optimize().
    f and k1 are bounded to [0.5, 2.0]x initial and [-1.0, 1.0] respectively
    to keep the solver on a physically plausible manifold.
    """
    camera_array = capture_volume.camera_array
    posed_cam_id_to_index = camera_array.posed_cam_id_to_index
    n_cams = len(posed_cam_id_to_index)
    cam_ids_by_index = sorted(posed_cam_id_to_index, key=lambda cid: posed_cam_id_to_index[cid])

    matched_mask = capture_volume.img_to_obj_map >= 0
    posed_cam_ids = set(posed_cam_id_to_index.keys())
    posed_mask: np.ndarray = capture_volume.image_points.df["cam_id"].isin(posed_cam_ids).to_numpy()
    combined_mask = matched_mask & posed_mask

    matched_img_df = capture_volume.image_points.df[combined_mask]

    camera_indices: CameraIndices = np.array(
        [posed_cam_id_to_index[cam_id] for cam_id in matched_img_df["cam_id"]],
        dtype=np.int16,
    )
    image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
    obj_indices = capture_volume.img_to_obj_map[combined_mask]

    cx_cy = np.zeros((n_cams, 2))
    dist_tail = np.zeros((n_cams, 4))  # k2, p1, p2, k3 (fixed)
    f_initial = np.zeros(n_cams)
    k1_initial = np.zeros(n_cams)
    camera_params = np.zeros((n_cams, N_CAM_PARAMS))

    for idx, cam_id in enumerate(cam_ids_by_index):
        cam = camera_array.cameras[cam_id]
        if cam.matrix is None or cam.distortions is None or cam.rotation is None or cam.translation is None:
            raise ValueError(f"Camera {cam_id} missing intrinsics or extrinsics for joint optimization")

        rvec = cv2.Rodrigues(cam.rotation)[0].ravel()
        f = float(cam.matrix[0, 0])
        k1 = float(cam.distortions[0])

        camera_params[idx, 0:3] = rvec
        camera_params[idx, 3:6] = cam.translation
        camera_params[idx, 6] = f
        camera_params[idx, 7] = k1

        cx_cy[idx] = cam.matrix[0, 2], cam.matrix[1, 2]
        dist_tail[idx] = cam.distortions[1:5]
        f_initial[idx] = f
        k1_initial[idx] = k1

    points_3d = capture_volume.world_points.points
    initial_params = np.concatenate([camera_params.ravel(), points_3d.ravel()])

    constraint_pairs = None
    constraint_distances = None
    constraint_weights = None

    if use_constraints and capture_volume.constraints is not None:
        arrays = capture_volume._build_constraint_arrays()
        if arrays is not None:
            constraint_pairs, constraint_distances, constraint_sigmas = arrays
            f_median = float(np.median(f_initial))
            constraint_weights = (pixel_sigma / f_median) / constraint_sigmas
            logger.info(f"Adding {len(constraint_pairs)} constraint rows (f_median={f_median:.0f})")

    n_points = len(points_3d)
    sparsity_pattern = _joint_sparsity_pattern(camera_indices, obj_indices, n_cams, n_points, constraint_pairs)

    lower_bounds = np.full_like(initial_params, -np.inf)
    upper_bounds = np.full_like(initial_params, np.inf)
    for idx in range(n_cams):
        f = f_initial[idx]
        lower_bounds[idx * N_CAM_PARAMS + 6] = 0.5 * f
        upper_bounds[idx * N_CAM_PARAMS + 6] = 2.0 * f
        lower_bounds[idx * N_CAM_PARAMS + 7] = -1.0
        upper_bounds[idx * N_CAM_PARAMS + 7] = 1.0

    n_obs = len(image_coords)
    logger.info(f"Beginning joint bundle adjustment on {n_obs} observations, {n_cams} cameras")
    result = least_squares(
        _joint_residuals,
        initial_params,
        args=(
            n_cams,
            camera_indices,
            image_coords,
            obj_indices,
            cx_cy,
            dist_tail,
            f_initial,
            constraint_pairs,
            constraint_distances,
            constraint_weights,
        ),
        jac_sparsity=sparsity_pattern,
        verbose=verbose,
        x_scale="jac",
        loss="linear",
        ftol=ftol,
        max_nfev=max_nfev,
        method="trf",
        bounds=(lower_bounds, upper_bounds),
    )

    termination_reason = _SCIPY_STATUS_REASONS.get(result.status, f"unknown_{result.status}")
    converged = result.status in (1, 2, 3, 4)

    if strict and not converged:
        from caliscope.exceptions import CalibrationError

        raise CalibrationError(
            f"Joint BA did not converge: {termination_reason}\n"
            f"Pass strict=False to suppress this error and inspect the result."
        )

    hit_bounds = False
    intrinsic_estimates: list[IntrinsicEstimate] = []
    for idx, cam_id in enumerate(cam_ids_by_index):
        f_rec = float(result.x[idx * N_CAM_PARAMS + 6])
        k1_rec = float(result.x[idx * N_CAM_PARAMS + 7])

        f_lo, f_hi = lower_bounds[idx * N_CAM_PARAMS + 6], upper_bounds[idx * N_CAM_PARAMS + 6]
        k1_lo, k1_hi = lower_bounds[idx * N_CAM_PARAMS + 7], upper_bounds[idx * N_CAM_PARAMS + 7]
        if abs(f_rec - f_lo) <= 0.01 * abs(f_lo) or abs(f_rec - f_hi) <= 0.01 * abs(f_hi):
            hit_bounds = True
        if abs(k1_rec - k1_lo) <= 0.01 or abs(k1_rec - k1_hi) <= 0.01:
            hit_bounds = True

        intrinsic_estimates.append(
            IntrinsicEstimate(
                cam_id=cam_id,
                f_recovered=f_rec,
                k1_recovered=k1_rec,
                f_initial=float(f_initial[idx]),
                k1_initial=float(k1_initial[idx]),
            )
        )

    new_camera_array = deepcopy(camera_array)
    for idx, cam_id in enumerate(cam_ids_by_index):
        rvec = result.x[idx * N_CAM_PARAMS : idx * N_CAM_PARAMS + 3]
        tvec = result.x[idx * N_CAM_PARAMS + 3 : idx * N_CAM_PARAMS + 6]
        new_camera_array.cameras[cam_id].rotation = cv2.Rodrigues(rvec)[0]
        new_camera_array.cameras[cam_id].translation = tvec

    optimized_points = result.x[n_cams * N_CAM_PARAMS :].reshape((-1, 3))
    new_world_df = capture_volume.world_points.df.copy()
    new_world_df[["x_coord", "y_coord", "z_coord"]] = optimized_points

    optimization_status = OptimizationStatus(
        converged=converged,
        termination_reason=termination_reason,
        iterations=result.nfev,
        final_cost=float(result.cost),
    )

    new_capture_volume = CaptureVolume(
        camera_array=new_camera_array,
        image_points=capture_volume.image_points,
        world_points=WorldPoints(new_world_df),
        constraints=capture_volume.constraints,
        _optimization_status=optimization_status,
    )

    return JointOptimizationResult(
        capture_volume=new_capture_volume,
        intrinsic_estimates=tuple(intrinsic_estimates),
        converged=converged,
        hit_bounds=hit_bounds,
        final_cost=float(result.cost),
    )
