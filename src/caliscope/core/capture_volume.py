from __future__ import annotations
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from copy import deepcopy
from numpy.typing import NDArray

import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal
import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.constraints import ConstraintSet, ConstraintViolation, RigidityReport
from caliscope.core.point_data import STATIC_SYNC_INDEX, ImagePoints, WorldPoints
from caliscope.core.reprojection import (
    ErrorsXY,
    reprojection_errors,
    bundle_residuals,
    ImageCoords,
    WorldCoords,
    CameraIndices,
)
from caliscope.core.reprojection_report import ReprojectionReport
from caliscope.core.alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
    SimilarityTransform,
)
from caliscope.core.scale_accuracy import (
    compute_frame_scale_error,
    FrameScaleError,
    VolumetricScaleReport,
)

import pandas as pd

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class OptimizationStatus:
    """Result metadata from bundle adjustment optimization.

    Populated by optimize(), cleared by filter methods.
    """

    converged: bool
    termination_reason: str  # "converged_gtol", "max_evaluations", etc.
    iterations: int  # nfev from scipy
    final_cost: float


# Mapping from scipy least_squares status codes to human-readable reasons
_SCIPY_STATUS_REASONS: dict[int, str] = {
    -1: "improper_input",
    0: "max_evaluations",
    1: "converged_gtol",
    2: "converged_ftol",
    3: "converged_xtol",
    4: "converged_small_step",
}


@dataclass(frozen=True)
class CaptureVolume:
    camera_array: CameraArray
    image_points: ImagePoints
    world_points: WorldPoints
    constraints: ConstraintSet | None = None
    # Computed field: maps each image observation to its world point index (-1 if unmatched)
    img_to_obj_map: np.ndarray = field(init=False)
    # Optimization metadata: None if capture volume hasn't been optimized or was filtered post-optimization
    _optimization_status: OptimizationStatus | None = field(default=None, compare=False)

    @property
    def optimization_status(self) -> OptimizationStatus | None:
        """Optimization result metadata, or None if not from optimize() call."""
        return self._optimization_status

    def __post_init__(self):
        """Compute mapping and validate geometry."""
        object.__setattr__(self, "img_to_obj_map", self._compute_img_to_obj_map())
        self._validate_geometry()

    def _validate_geometry(self):
        """Ensure data counts make geometric sense."""
        n_img = len(self.image_points.df)
        n_world = len(self.world_points.df)
        n_cams = len(self.camera_array.posed_cameras)

        if n_img == 0:
            raise ValueError("No image observations provided")
        if n_world == 0:
            raise ValueError("No world points provided")
        if n_cams == 0:
            raise ValueError("No posed cameras in array")

        # Check that we have at least some matched observations
        n_matched = np.sum(self.img_to_obj_map >= 0)
        if n_matched == 0:
            raise ValueError("No image observations have corresponding world points")

        if n_matched < n_world * 2:
            logger.warning(
                f"Suspicious geometry: {n_matched} matched observations for {n_world} world points. "
                f"Expected at least {n_world * 2} for multi-view geometry."
            )
        # Validate indices are in bounds
        valid_indices = self.img_to_obj_map[self.img_to_obj_map >= 0]
        if valid_indices.size > 0 and valid_indices.max() >= n_world:
            raise ValueError(f"obj_indices contains out-of-bounds index: {valid_indices.max()} >= {n_world}")

    def _compute_img_to_obj_map(self) -> np.ndarray:
        """Map each image observation to its world point index. Returns -1 for unmatched."""
        world_df = self.world_points.df.reset_index().rename(columns={"index": "world_idx"})
        mapping = world_df.set_index(["sync_index", "object_id", "keypoint_id"])["world_idx"].to_dict()

        static_ids = self.constraints.static_object_ids if self.constraints else frozenset()

        img_df = self.image_points.df
        keys = []
        for sync_idx, obj_id, kp_id in zip(img_df["sync_index"], img_df["object_id"], img_df["keypoint_id"]):
            if int(obj_id) in static_ids:
                keys.append((STATIC_SYNC_INDEX, int(obj_id), int(kp_id)))
            else:
                keys.append((int(sync_idx), int(obj_id), int(kp_id)))
        img_to_obj_map = np.array([mapping.get(key, -1) for key in keys], dtype=np.int32)

        n_unmatched = np.sum(img_to_obj_map == -1)
        if n_unmatched > 0:
            logger.info(f"{n_unmatched} of {len(img_to_obj_map)} image observations have no world point")

        return img_to_obj_map

    @cached_property
    def reprojection_report(self) -> ReprojectionReport:
        """
        Generate comprehensive reprojection error report in pixel units.
        Cached automatically since capture volume data is immutable.
        """
        # 1. Filter to matched observations from posed cameras only
        matched_mask = self.img_to_obj_map >= 0
        posed_cam_ids = set(self.camera_array.posed_cam_id_to_index.keys())
        posed_mask: np.ndarray = self.image_points.df["cam_id"].isin(posed_cam_ids).to_numpy()
        combined_mask = matched_mask & posed_mask

        n_total = len(self.img_to_obj_map)
        n_matched = combined_mask.sum()
        n_unmatched = n_total - n_matched

        if n_matched == 0:
            raise ValueError("No matched observations for reprojection error calculation")

        matched_img_df = self.image_points.df[combined_mask]
        matched_obj_indices = self.img_to_obj_map[combined_mask]

        # 2. Prepare arrays for core function
        camera_indices: CameraIndices = np.array(
            [self.camera_array.posed_cam_id_to_index[cam_id] for cam_id in matched_img_df["cam_id"]], dtype=np.int16
        )
        image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
        world_coords: WorldCoords = self.world_points.points[matched_obj_indices]

        # 3. Compute reprojection errors
        errors_xy: ErrorsXY = reprojection_errors(
            self.camera_array, camera_indices, image_coords, world_coords, use_normalized=False
        )

        # 4. Build raw_errors DataFrame
        euclidean_error = np.sqrt(np.sum(errors_xy**2, axis=1))
        raw_errors = pd.DataFrame(
            {
                "sync_index": matched_img_df["sync_index"].values,
                "cam_id": matched_img_df["cam_id"].values,
                "object_id": matched_img_df["object_id"].values,
                "keypoint_id": matched_img_df["keypoint_id"].values,
                "error_x": errors_xy[:, 0],
                "error_y": errors_xy[:, 1],
                "euclidean_error": euclidean_error,
            }
        )

        # 5. Aggregate metrics
        overall_rmse = float(np.sqrt(np.mean(euclidean_error**2)))

        by_camera = {}
        for cam_id in self.camera_array.posed_cameras.keys():
            cam_errors = euclidean_error[matched_img_df["cam_id"] == cam_id]
            by_camera[cam_id] = float(np.sqrt(np.mean(cam_errors**2))) if len(cam_errors) > 0 else 0.0

        by_point = {}
        # Build composite key for per-point RMSE
        point_keys = list(zip(matched_img_df["object_id"], matched_img_df["keypoint_id"]))
        unique_keys = set(point_keys)
        for obj_id, kp_id in unique_keys:
            mask = (matched_img_df["object_id"].values == obj_id) & (matched_img_df["keypoint_id"].values == kp_id)
            point_errors = euclidean_error[mask]
            by_point[(obj_id, kp_id)] = float(np.sqrt(np.mean(point_errors**2)))

        # 6. Count unmatched by camera (only count for posed cameras)
        unmatched_by_camera = {}
        for cam_id in self.camera_array.cameras.keys():
            cam_total = (self.image_points.df["cam_id"] == cam_id).sum()
            cam_matched = ((self.image_points.df["cam_id"] == cam_id) & combined_mask).sum()
            unmatched_by_camera[cam_id] = int(cam_total - cam_matched)

        # 7. Create and cache report
        report = ReprojectionReport(
            overall_rmse=overall_rmse,
            by_camera=by_camera,
            by_point=by_point,
            n_unmatched_observations=int(n_unmatched),
            unmatched_rate=n_unmatched / n_total if n_total > 0 else 0.0,
            unmatched_by_camera=unmatched_by_camera,
            raw_errors=raw_errors,
            n_observations_matched=int(n_matched),
            n_observations_total=int(n_total),
            n_cameras=len(self.camera_array.posed_cameras),
            n_points=len(self.world_points.points),
        )

        return report

    def save(self, directory: Path | str) -> None:
        """Save capture volume to a directory.

        Writes camera_array.toml, image_points.csv, world_points.csv.
        Note: optimization_status is not persisted.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.camera_array.to_toml(directory / "camera_array.toml")
        self.image_points.to_csv(directory / "image_points.csv")
        self.world_points.to_csv(directory / "world_points.csv")
        if self.constraints is not None:
            self.constraints.to_toml(directory / "constraints.toml")

    @classmethod
    def load(cls, directory: Path | str) -> CaptureVolume:
        """Load capture volume from a directory.

        Expects: camera_array.toml, image_points.csv, world_points.csv.
        Optionally loads constraints.toml if present.
        """
        directory = Path(directory)
        camera_array = CameraArray.from_toml(directory / "camera_array.toml")
        image_points = ImagePoints.from_csv(directory / "image_points.csv")
        world_points = WorldPoints.from_csv(directory / "world_points.csv")
        constraints_path = directory / "constraints.toml"
        constraints = ConstraintSet.from_toml(constraints_path) if constraints_path.exists() else None
        return cls(
            camera_array=camera_array, image_points=image_points, world_points=world_points, constraints=constraints
        )

    @classmethod
    def bootstrap(
        cls,
        image_points: ImagePoints,
        camera_array: CameraArray,
        constraints: ConstraintSet | None = None,
    ) -> CaptureVolume:
        """Bootstrap extrinsic calibration from 2D observations.

        Pipeline: deepcopy cameras → build pose network → apply poses → triangulate.
        Does NOT auto-optimize. Call .optimize() on the result.
        The input CameraArray is not modified.

        Raises:
            CalibrationError: If cameras lack intrinsics, cam_ids mismatch,
                or insufficient stereo pairs exist.
        """
        from caliscope.exceptions import CalibrationError
        from caliscope.core.bootstrap_pose.build_paired_pose_network import (
            build_paired_pose_network,
        )

        # Validate: cam_id mismatch
        point_cam_ids = set(image_points.df["cam_id"].unique())
        array_cam_ids = set(camera_array.cameras.keys())
        missing_cameras = point_cam_ids - array_cam_ids
        if missing_cameras:
            raise CalibrationError(f"ImagePoints reference cameras {missing_cameras} not in the CameraArray.")

        # Validate: intrinsics
        uncalibrated = [
            cam_id for cam_id, cam in camera_array.cameras.items() if cam.matrix is None or cam.distortions is None
        ]
        if uncalibrated:
            raise CalibrationError(
                f"Cannot run extrinsic calibration -- cameras {uncalibrated} have "
                f"no intrinsic calibration.\n\n"
                f"Run calibrate_intrinsics() for each camera first:\n"
                f"    output = calibrate_intrinsics(points, cameras[{uncalibrated[0]}])\n"
                f"    cameras[{uncalibrated[0]}] = output.camera"
            )

        # Validate: obj_loc presence
        obj_cols = image_points.df[["obj_loc_x", "obj_loc_y", "obj_loc_z"]]
        if obj_cols.isna().all().all():
            raise CalibrationError(
                "ImagePoints contain no object location data (obj_loc columns are all NaN). "
                "Extrinsic calibration requires a tracker that provides known 3D positions "
                "(e.g., CharucoTracker)."
            )

        cameras = deepcopy(camera_array)
        pose_network = build_paired_pose_network(image_points, cameras)
        pose_network.apply_to(cameras)
        static_ids = constraints.static_object_ids if constraints else frozenset()
        world_points = image_points.triangulate(cameras, static_object_ids=static_ids)

        return cls(camera_array=cameras, image_points=image_points, world_points=world_points, constraints=constraints)

    def optimize(
        self,
        ftol: float = 1e-8,
        max_nfev: int = 1000,
        verbose: int = 0,
        strict: bool = True,
        use_constraints: bool = True,
        pixel_sigma: float = 1.0,
    ) -> CaptureVolume:
        """Perform bundle adjustment optimization on this CaptureVolume.

        Returns a NEW CaptureVolume with optimized camera parameters and 3D points.
        The original remains unchanged (immutable pattern).
        """
        matched_mask = self.img_to_obj_map >= 0
        posed_cam_ids = set(self.camera_array.posed_cam_id_to_index.keys())
        posed_mask: np.ndarray = self.image_points.df["cam_id"].isin(posed_cam_ids).to_numpy()
        combined_mask = matched_mask & posed_mask

        matched_img_df = self.image_points.df[combined_mask]

        camera_indices: CameraIndices = np.array(
            [self.camera_array.posed_cam_id_to_index[cam_id] for cam_id in matched_img_df["cam_id"]], dtype=np.int16
        )

        image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
        image_to_world_indices = self.img_to_obj_map[combined_mask]

        initial_params = self._get_vectorized_params()

        # Build constraint arrays if available
        constraint_pairs = None
        constraint_distances = None
        constraint_weights = None

        if use_constraints and self.constraints is not None:
            arrays = self._build_constraint_arrays()
            if arrays is not None:
                constraint_pairs, constraint_distances, constraint_sigmas = arrays
                # w = (pixel_sigma / f_median) / sigma_c
                # A sigma_c-sized violation costs the same as a pixel_sigma-sized
                # reprojection error. Collapses to 1/sigma_c under Phase 2 whitened residuals.
                # posed_cameras guarantees matrix is not None
                focal_lengths = [
                    cam.matrix[0, 0] for cam in self.camera_array.posed_cameras.values() if cam.matrix is not None
                ]
                f_median = float(np.median(focal_lengths))
                constraint_weights = (pixel_sigma / f_median) / constraint_sigmas
                n_c = len(constraint_pairs)
                logger.info(f"Adding {n_c} constraint rows (f_median={f_median:.0f}, pixel_sigma={pixel_sigma})")

        sparsity_pattern = self._get_sparsity_pattern(camera_indices, image_to_world_indices, constraint_pairs)

        n_obs = len(image_coords)
        logger.info(f"Beginning bundle adjustment on {n_obs} observations")
        result = least_squares(
            bundle_residuals,
            initial_params,
            args=(
                self.camera_array,
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
            loss="linear",
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
                f"Bundle adjustment did not converge: {termination_reason}\n"
                f"Pass strict=False to suppress this error and inspect the result."
            )

        new_camera_array = deepcopy(self.camera_array)
        new_camera_array.update_extrinsic_params(result.x)

        n_cams = len(self.camera_array.posed_cameras)
        n_cam_params = 6
        optimized_points = result.x[n_cams * n_cam_params :].reshape((-1, 3))

        new_world_df = self.world_points.df.copy()
        new_world_df[["x_coord", "y_coord", "z_coord"]] = optimized_points

        new_world_points = WorldPoints(new_world_df)

        return CaptureVolume(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
            constraints=self.constraints,
            _optimization_status=optimization_status,
        )

    def _get_sparsity_pattern(
        self,
        camera_indices: NDArray[np.int16],
        obj_indices: NDArray[np.int32],
        constraint_pairs: NDArray[np.int32] | None = None,
    ) -> lil_matrix:
        """Generate sparsity pattern for Jacobian matrix.

        Reprojection rows: each depends on 6 camera params + 3 point params.
        Constraint rows: each depends on 6 point params (3 per endpoint),
        no camera params.
        """
        n_observations = len(camera_indices)
        n_cameras = len(self.camera_array.posed_cameras)
        n_points = len(self.world_points.points)

        n_constraints = len(constraint_pairs) if constraint_pairs is not None else 0
        n_residuals = n_observations * 2 + n_constraints
        n_params = n_cameras * 6 + n_points * 3

        sparsity = lil_matrix((n_residuals, n_params), dtype=int)

        obs_idx = np.arange(n_observations)

        for cam_param in range(6):
            param_col = camera_indices * 6 + cam_param
            sparsity[2 * obs_idx, param_col] = 1
            sparsity[2 * obs_idx + 1, param_col] = 1

        for point_param in range(3):
            param_col = n_cameras * 6 + obj_indices * 3 + point_param
            sparsity[2 * obs_idx, param_col] = 1
            sparsity[2 * obs_idx + 1, param_col] = 1

        if constraint_pairs is not None:
            c_idx = np.arange(n_constraints)
            row_offset = n_observations * 2
            for coord in range(3):
                col_a = n_cameras * 6 + constraint_pairs[:, 0] * 3 + coord
                col_b = n_cameras * 6 + constraint_pairs[:, 1] * 3 + coord
                sparsity[row_offset + c_idx, col_a] = 1
                sparsity[row_offset + c_idx, col_b] = 1

        return sparsity

    def _get_vectorized_params(self) -> NDArray[np.float64]:
        """
        Convert camera extrinsics and 3D points to a flattened optimization vector.
        Shape: (n_cameras*6 + n_points*3,)
        """
        camera_params = self.camera_array.get_extrinsic_params()  # (n_cams, 6)
        if camera_params is None:
            raise ValueError("Camera extrinsic parameters not initialized")
        points_3d = self.world_points.points  # (n_points, 3)

        return np.concatenate([camera_params.ravel(), points_3d.ravel()])

    def _build_constraint_arrays(
        self,
    ) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]] | None:
        """Build constraint arrays for the BA from self.constraints.

        Returns (pair_indices (n_c, 2), distances (n_c,), sigmas (n_c,))
        where pair_indices are row indices into world_points.df.
        Returns None if no constraints or no valid instances.
        """
        if self.constraints is None or not self.constraints.distances:
            return None

        from collections import defaultdict

        world_df = self.world_points.df
        static_ids = self.constraints.static_object_ids

        # (object_id, keypoint_id) -> {sync_index: row_idx}
        point_lookup: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        for row_idx, (si, oid, kid) in enumerate(
            zip(world_df["sync_index"], world_df["object_id"], world_df["keypoint_id"])
        ):
            point_lookup[(int(oid), int(kid))][int(si)] = row_idx

        pairs: list[list[int]] = []
        dists: list[float] = []
        sigmas: list[float] = []

        for dc in self.constraints.distances:
            key_a = (dc.object_id_a, dc.keypoint_id_a)
            key_b = (dc.object_id_b, dc.keypoint_id_b)
            a_static = dc.object_id_a in static_ids
            b_static = dc.object_id_b in static_ids

            if a_static != b_static:
                continue

            syncs_a = point_lookup.get(key_a, {})
            syncs_b = point_lookup.get(key_b, {})

            if a_static:
                # Both static: one instance at STATIC_SYNC_INDEX
                if STATIC_SYNC_INDEX in syncs_a and STATIC_SYNC_INDEX in syncs_b:
                    pairs.append([syncs_a[STATIC_SYNC_INDEX], syncs_b[STATIC_SYNC_INDEX]])
                    dists.append(dc.distance)
                    sigmas.append(dc.sigma)
            else:
                # Both mobile: one instance per shared sync_index
                shared = syncs_a.keys() & syncs_b.keys()
                for si in shared:
                    if si == STATIC_SYNC_INDEX:
                        continue
                    pairs.append([syncs_a[si], syncs_b[si]])
                    dists.append(dc.distance)
                    sigmas.append(dc.sigma)

        if not pairs:
            return None

        return (
            np.array(pairs, dtype=np.int32),
            np.array(dists, dtype=np.float64),
            np.array(sigmas, dtype=np.float64),
        )

    def rigidity_report(self) -> RigidityReport:
        """Measure constraint violations against current world points.

        Pure measurement, no optimization. Valid on any CaptureVolume with
        constraints, before or after optimize().
        """
        if self.constraints is None or not self.constraints.distances:
            return RigidityReport(violations=())

        from collections import defaultdict

        world_df = self.world_points.df
        static_ids = self.constraints.static_object_ids
        coords = self.world_points.points

        point_lookup: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        for row_idx, (si, oid, kid) in enumerate(
            zip(world_df["sync_index"], world_df["object_id"], world_df["keypoint_id"])
        ):
            point_lookup[(int(oid), int(kid))][int(si)] = row_idx

        violations: list[ConstraintViolation] = []

        for dc in self.constraints.distances:
            key_a = (dc.object_id_a, dc.keypoint_id_a)
            key_b = (dc.object_id_b, dc.keypoint_id_b)
            a_static = dc.object_id_a in static_ids
            b_static = dc.object_id_b in static_ids

            if a_static != b_static:
                continue

            syncs_a = point_lookup.get(key_a, {})
            syncs_b = point_lookup.get(key_b, {})

            if a_static:
                if STATIC_SYNC_INDEX in syncs_a and STATIC_SYNC_INDEX in syncs_b:
                    actual = float(
                        np.linalg.norm(coords[syncs_a[STATIC_SYNC_INDEX]] - coords[syncs_b[STATIC_SYNC_INDEX]])
                    )
                    violations.append(
                        ConstraintViolation(
                            object_id_a=dc.object_id_a,
                            keypoint_id_a=dc.keypoint_id_a,
                            object_id_b=dc.object_id_b,
                            keypoint_id_b=dc.keypoint_id_b,
                            sync_index=STATIC_SYNC_INDEX,
                            expected=dc.distance,
                            actual=actual,
                        )
                    )
            else:
                shared = syncs_a.keys() & syncs_b.keys()
                for si in shared:
                    if si == STATIC_SYNC_INDEX:
                        continue
                    actual = float(np.linalg.norm(coords[syncs_a[si]] - coords[syncs_b[si]]))
                    violations.append(
                        ConstraintViolation(
                            object_id_a=dc.object_id_a,
                            keypoint_id_a=dc.keypoint_id_a,
                            object_id_b=dc.object_id_b,
                            keypoint_id_b=dc.keypoint_id_b,
                            sync_index=si,
                            expected=dc.distance,
                            actual=actual,
                        )
                    )

        return RigidityReport(violations=tuple(violations))

    def _filter_by_reprojection_thresholds(self, thresholds: dict[int, float], min_per_camera: int) -> CaptureVolume:
        """
        Internal: Filter observations using per-camera error thresholds with safety enforcement.

        Args:
            thresholds: dict mapping camera cam_id -> max_error_pixels for that camera
            min_per_camera: minimum observations to preserve per camera

        Returns:
            New CaptureVolume with filtered observations
        """
        # Get reprojection data (cached)
        report = self.reprojection_report
        raw_errors = report.raw_errors

        # Build initial keep mask: error <= threshold for that camera's cam_id
        threshold_series = raw_errors["cam_id"].map(thresholds)
        keep_mask = (raw_errors["euclidean_error"] <= threshold_series).copy()

        # Apply safety: ensure each camera keeps at least min_per_camera observations
        for cam_id in raw_errors["cam_id"].unique():
            camera_idx = raw_errors["cam_id"] == cam_id
            n_keep = keep_mask[camera_idx].sum()
            n_total = camera_idx.sum()

            # If below minimum and we can add more
            if n_keep < min_per_camera and n_keep < n_total:
                # How many we need to add (capped at total available)
                n_needed = min(min_per_camera, n_total) - n_keep

                # Get errors for observations that would be filtered out
                # pandas stubs don't narrow .loc with boolean mask to Series
                filtered_errors = raw_errors.loc[camera_idx & ~keep_mask, "euclidean_error"]

                if len(filtered_errors) >= n_needed:  # type: ignore[arg-type]
                    # Find the error threshold that would keep exactly n_needed more observations
                    threshold_to_add: float = filtered_errors.nsmallest(n_needed).iloc[-1]  # type: ignore[union-attr]

                    # Update mask to keep observations with error <= threshold_to_add
                    keep_mask[camera_idx] = raw_errors.loc[camera_idx, "euclidean_error"] <= threshold_to_add  # type: ignore[index, operator]

        # Get keys of observations to keep
        keep_keys = raw_errors[keep_mask][["sync_index", "cam_id", "object_id", "keypoint_id"]]

        # Filter image points by merging with keep keys
        filtered_img_df = self.image_points.df.merge(
            keep_keys, on=["sync_index", "cam_id", "object_id", "keypoint_id"], how="inner"
        )
        filtered_image_points = ImagePoints(filtered_img_df)

        # Prune orphaned world points (3D points with no observations)
        remaining_3d_keys = filtered_img_df[["sync_index", "object_id", "keypoint_id"]].drop_duplicates()
        filtered_world_df = self.world_points.df.merge(
            remaining_3d_keys, on=["sync_index", "object_id", "keypoint_id"], how="inner"
        )

        filtered_world_points = WorldPoints(filtered_world_df)

        return CaptureVolume(
            camera_array=self.camera_array,
            image_points=filtered_image_points,
            world_points=filtered_world_points,
            constraints=self.constraints,
        )

    def filter_by_absolute_error(self, max_pixels: float, min_per_camera: int = 10) -> CaptureVolume:
        """
        Remove observations with reprojection error > max_pixels.

        Safety: Ensures each camera keeps at least min_per_camera observations.
        If a camera would drop below this threshold, the lowest-error observations
        are restored until the threshold is met.

        Args:
            max_pixels: Maximum reprojection error (pixels) to keep
            min_per_camera: Minimum observations per camera (safety floor)

        Returns:
            New CaptureVolume with filtered observations
        """
        if max_pixels <= 0:
            raise ValueError(f"max_pixels must be positive, got {max_pixels}")

        if min_per_camera < 1:
            raise ValueError(f"min_per_camera must be >= 1, got {min_per_camera}")

        # Build uniform thresholds for all posed cameras
        thresholds = {cam_id: max_pixels for cam_id in self.camera_array.posed_cameras.keys()}

        return self._filter_by_reprojection_thresholds(thresholds, min_per_camera)

    def filter_by_percentile_error(
        self, percentile: float, scope: Literal["per_camera", "overall"] = "per_camera", min_per_camera: int = 10
    ) -> CaptureVolume:
        """
        Remove worst N% of observations based on reprojection error.

        Args:
            percentile: Percentage of worst observations to remove (0-100)
            scope: "per_camera" computes percentile per camera, "overall" uses global percentile
            min_per_camera: Minimum observations per camera (safety floor)

        Returns:
            New CaptureVolume with filtered observations
        """
        if not (0 < percentile <= 100):
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

        if min_per_camera < 1:
            raise ValueError(f"min_per_camera must be >= 1, got {min_per_camera}")

        report = self.reprojection_report
        raw_errors = report.raw_errors

        if scope == "per_camera":
            # Compute (100 - percentile)th percentile per camera
            thresholds: dict[int, float] = {}
            for cam_id in self.camera_array.posed_cameras.keys():
                camera_errors = raw_errors[raw_errors["cam_id"] == cam_id]["euclidean_error"]
                if len(camera_errors) > 0:
                    # Keep the best (100 - percentile) percent
                    keep_percentile = 100 - percentile
                    thresholds[cam_id] = float(np.percentile(camera_errors, keep_percentile))
                else:
                    thresholds[cam_id] = float(np.inf)  # No observations, keep nothing

        elif scope == "overall":
            # Compute global (100 - percentile)th percentile
            keep_percentile = 100 - percentile
            global_threshold = float(np.percentile(raw_errors["euclidean_error"], keep_percentile))
            thresholds = {cam_id: global_threshold for cam_id in self.camera_array.posed_cameras.keys()}

        else:
            raise ValueError(f"scope must be 'per_camera' or 'overall', got {scope}")

        return self._filter_by_reprojection_thresholds(thresholds, min_per_camera)

    def compute_volumetric_scale_accuracy(self) -> VolumetricScaleReport:
        """Compute per-marker, per-frame scale accuracy across the capture volume.

        Groups by (sync_index, object_id) so each marker's pairwise distances
        are compared against its own local-frame geometry. Cross-marker distances
        are never computed (each marker's obj_loc is in its own coordinate frame).

        Returns:
            VolumetricScaleReport containing per-frame-per-object errors and aggregate metrics.
            Returns empty report if no valid frames exist (normal pre-alignment state).
        """
        img_df = self.image_points.df
        world_df = self.world_points.df

        obj_loc_cols = ["obj_loc_x", "obj_loc_y", "obj_loc_z"]
        if not all(col in img_df.columns for col in obj_loc_cols):
            return VolumetricScaleReport.empty()

        obj_loc_mask = ~img_df[["obj_loc_x", "obj_loc_y"]].isna().any(axis=1)
        img_with_obj = img_df[obj_loc_mask]

        if img_with_obj.empty:
            return VolumetricScaleReport.empty()

        static_ids = self.constraints.static_object_ids if self.constraints else frozenset()

        frame_errors: list[FrameScaleError] = []

        for (sync_index_raw, object_id_raw), img_group in img_with_obj.groupby(["sync_index", "object_id"]):
            sync_index = int(sync_index_raw)  # type: ignore[arg-type]
            object_id = int(object_id_raw)  # type: ignore[arg-type]

            # Static markers: world points live at STATIC_SYNC_INDEX
            world_si = STATIC_SYNC_INDEX if object_id in static_ids else sync_index
            world_subset = world_df[(world_df["sync_index"] == world_si) & (world_df["object_id"] == object_id)]

            if world_subset.empty:
                continue

            obj_points_df = img_group[
                ["object_id", "keypoint_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]
            ].drop_duplicates(subset=["object_id", "keypoint_id"])

            merged = world_subset.merge(obj_points_df, on=["object_id", "keypoint_id"], how="inner")

            if merged["obj_loc_z"].isna().all():
                merged = merged.copy()
                merged["obj_loc_z"] = 0.0

            valid_mask = ~merged[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].isna().any(axis=1)
            merged = merged[valid_mask]

            if len(merged) < 3:
                continue

            n_cameras_contributing = int(img_group["cam_id"].nunique())

            world_points = merged[["x_coord", "y_coord", "z_coord"]].to_numpy()
            object_points = merged[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].to_numpy()

            try:
                frame_error = compute_frame_scale_error(
                    world_points=world_points,
                    object_points=object_points,
                    sync_index=sync_index,
                    object_id=object_id,
                    n_cameras_contributing=n_cameras_contributing,
                )
                frame_errors.append(frame_error)
            except ValueError as e:
                logger.debug(f"Skipping sync_index {sync_index} object_id {object_id}: {e}")
                continue

        return VolumetricScaleReport(frame_errors=tuple(frame_errors))

    def align_to_object(self, sync_index: int) -> "CaptureVolume":
        """
        Align the capture volume to real-world units using object point correspondences.

        Uses the 3D points triangulated at the given sync_index and their
        corresponding ground truth object positions (from obj_loc columns) to
        estimate a similarity transform that scales the reconstruction to real-world units.

        Note:
            Object coordinates (obj_loc_*) must be in real-world units (typically meters).
            For Charuco boards, this requires defining the board with square_length in meters.

        For planar Charuco boards, obj_loc_z may be missing and will be treated as 0.
        The obj_loc coordinates must be in the target units (typically meters).

        Args:
            sync_index: Frame index where object is visible and has obj_loc data

        Returns:
            New CaptureVolume with cameras and world points in object coordinate units

        Raises:
            ValueError: If insufficient valid correspondences (< 3 points) or missing data
        """
        # Extract data at sync_index
        img_df = self.image_points.df
        world_df = self.world_points.df

        img_subset = img_df[img_df["sync_index"] == sync_index]
        world_subset = world_df[world_df["sync_index"] == sync_index]

        if img_subset.empty:
            raise ValueError(f"No image observations at sync_index {sync_index}")
        if world_subset.empty:
            raise ValueError(f"No world points at sync_index {sync_index}")

        # Multi-marker guard: similarity transform between world points (global frame)
        # and obj_loc (mixed local frames) is geometrically meaningless
        unique_objects = img_subset["object_id"].unique()
        if len(unique_objects) > 1:
            raise ValueError(
                f"align_to_object requires single-object data at sync_index {sync_index}, "
                f"got object_ids {sorted(unique_objects)}. "
                "Multi-marker alignment requires Branch 3 constraint file."
            )

        # Merge on (object_id, keypoint_id) to find correspondences
        merged = pd.merge(
            world_subset[["object_id", "keypoint_id", "x_coord", "y_coord", "z_coord"]],
            img_subset[["object_id", "keypoint_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]],
            on=["object_id", "keypoint_id"],
            how="inner",
        )

        if len(merged) < 3:
            raise ValueError(f"Need at least 3 point correspondences at sync_index {sync_index}, got {len(merged)}")

        # Handle missing obj_loc_z (planar boards)
        if merged["obj_loc_z"].isna().all():
            logger.info("obj_loc_z is all NaN, assuming planar board with z=0")
            merged["obj_loc_z"] = 0.0

        # Filter out any rows with NaN object coordinates
        obj_cols = ["obj_loc_x", "obj_loc_y", "obj_loc_z"]
        valid_mask = ~merged[obj_cols].isna().any(axis=1)
        merged = merged[valid_mask]

        if len(merged) < 3:
            raise ValueError(
                f"Need at least 3 valid point correspondences at sync_index {sync_index}, "
                f"got {len(merged)} after filtering NaN values"
            )

        # Prepare source (triangulated) and target (object) points
        source_points = merged[["x_coord", "y_coord", "z_coord"]].values.astype(np.float64)
        target_points = merged[obj_cols].values.astype(np.float64)

        # Estimate and apply transform
        transform = estimate_similarity_transform(source_points, target_points)

        logger.info(
            f"Estimated alignment: scale={transform.scale:.6f}, "
            f"translation={transform.translation}, rotation_det={np.linalg.det(transform.rotation):.6f}"
        )

        new_camera_array, new_world_points = apply_similarity_transform(self.camera_array, self.world_points, transform)

        return CaptureVolume(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
            constraints=self.constraints,
            _optimization_status=self._optimization_status,
        )

    @property
    def unique_sync_indices(self) -> np.ndarray:
        """
        Return sorted array of unique sync_index values present in world_points.

        Used for slider range in visualization widgets.
        """
        indices = self.world_points.df["sync_index"].unique()
        return np.sort(indices)

    def rotate(self, axis: Literal["x", "y", "z"], angle_degrees: float) -> "CaptureVolume":
        """
        Rotate the coordinate system around the specified axis.

        Uses right-hand rule: positive angle = counter-clockwise rotation
        when looking down the positive axis toward the origin.

        Transforms both camera extrinsics and world points, returning a new
        immutable CaptureVolume. The original remains unchanged.

        Args:
            axis: The axis to rotate around ("x", "y", or "z")
            angle_degrees: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            New CaptureVolume with rotated coordinate system.
        """
        angle_rad = np.radians(angle_degrees)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # Standard rotation matrices following right-hand rule
        if axis == "x":
            rotation = np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ],
                dtype=np.float64,
            )
        elif axis == "y":
            rotation = np.array(
                [
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c],
                ],
                dtype=np.float64,
            )
        elif axis == "z":
            rotation = np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")

        transform = SimilarityTransform(
            rotation=rotation,
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
        )

        new_camera_array, new_world_points = apply_similarity_transform(self.camera_array, self.world_points, transform)

        return CaptureVolume(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
            constraints=self.constraints,
            _optimization_status=self._optimization_status,
        )


if __name__ == "__main__":
    from pathlib import Path
    from caliscope import __root__
    from caliscope.core.point_data import ImagePoints
    from caliscope.core.capture_volume import CaptureVolume
    from caliscope.cameras.camera_array import CameraArray

    # Load test data
    session_path = Path(__root__, "tests", "sessions", "larger_calibration_post_monocal")
    xy_path = session_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
    array_path = session_path / "camera_array.toml"

    image_points = ImagePoints.from_csv(xy_path)
    camera_array = CameraArray.from_toml(array_path)
    world_points = image_points.triangulate(camera_array)

    capture_volume = CaptureVolume(camera_array, image_points, world_points)

    # Inspect the reprojection report
    report = capture_volume.reprojection_report
