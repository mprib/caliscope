from __future__ import annotations
from scipy.optimize import least_squares
from copy import deepcopy
from numpy.typing import NDArray

import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal
import logging
import warnings

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.constraints import ConstraintSet, ConstraintViolation, RigidityReport
from caliscope.core.point_data import STATIC_SYNC_INDEX, ImagePoints, WorldPoints
from caliscope.core.reprojection import (
    ErrorsXY,
    reprojection_errors,
    joint_residuals,
    joint_jacobian,
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
from caliscope.core.coordinate_frame import world_basis_from_up_and_forward
from caliscope.core.scale_cues import CameraDistance, DepthObservation, SegmentLength

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
    bound_warnings: tuple = ()


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

    def pixel_f_scale(self, px: float = 1.0) -> float:
        """Convert a pixel inlier threshold to the normalized residual space.

        Residuals are divided by fx_initial per camera. f_scale = px / f_median
        maps a 1-pixel threshold to the same units.
        """
        focal_lengths = [cam.matrix[0, 0] for cam in self.camera_array.posed_cameras.values() if cam.matrix is not None]
        return px / float(np.median(focal_lengths))

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
        errors_xy: ErrorsXY = reprojection_errors(self.camera_array, camera_indices, image_coords, world_coords)

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

        # obj_loc presence is no longer validated here: build_paired_pose_network
        # branches on it, using the PnP path when object geometry is present and
        # the essential-matrix (epipolar) path when obj_loc is all NaN.
        cameras = deepcopy(camera_array)
        pose_network = build_paired_pose_network(image_points, cameras)
        pose_network.apply_to(cameras)
        static_ids = constraints.static_object_ids if constraints else frozenset()
        world_points = image_points.triangulate(cameras, static_object_ids=static_ids)

        return cls(camera_array=cameras, image_points=image_points, world_points=world_points, constraints=constraints)

    def optimize(
        self,
        ftol: float = 1e-8,
        max_nfev: int | None = None,
        verbose: int = 0,
        strict: bool = True,
        use_constraints: bool = True,
        pixel_sigma: float = 1.0,
        *,
        refine_intrinsics: bool = False,
        loss: str = "linear",
        f_scale: float = 1.0,
    ) -> CaptureVolume:
        """Bundle adjustment via pixel-space residuals.

        Extrinsics-only by default. Pass refine_intrinsics=True for joint
        intrinsic recovery (the production calibration workflow does).

        Free intrinsics converge slowly when depth variation is poor or
        constraints are absent — f and scale are coupled without a metric
        anchor. The depth-ratio metric characterizes this risk.
        """
        from caliscope.core.bundle_parameterization import BundleParameterization

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

        new_camera_array = deepcopy(self.camera_array)

        parameterization = BundleParameterization.from_camera_array(
            new_camera_array, n_points=len(self.world_points.points), refine_intrinsics=refine_intrinsics
        )
        x0 = parameterization.pack(new_camera_array, self.world_points.points)

        # Build constraint arrays if available
        constraint_groups_a = None
        constraint_groups_b = None
        constraint_distances = None
        constraint_weights = None

        if use_constraints and self.constraints is not None:
            arrays = self._build_constraint_arrays()
            if arrays is not None:
                constraint_groups_a, constraint_groups_b, constraint_distances, constraint_sigmas = arrays
                focal_lengths = [
                    cam.matrix[0, 0] for cam in self.camera_array.posed_cameras.values() if cam.matrix is not None
                ]
                f_median = float(np.median(focal_lengths))
                constraint_weights = (pixel_sigma / f_median) / constraint_sigmas
                n_c = len(constraint_groups_a)
                logger.info(f"Adding {n_c} constraint rows (f_median={f_median:.0f}, pixel_sigma={pixel_sigma})")

        n_obs = len(image_coords)
        logger.info(f"Beginning bundle adjustment on {n_obs} observations")
        result = least_squares(
            joint_residuals,
            x0,
            args=(
                parameterization,
                camera_indices,
                image_coords,
                image_to_world_indices,
                constraint_groups_a,
                constraint_groups_b,
                constraint_distances,
                constraint_weights,
            ),
            # scipy's stubs type jac as the str literals only; a callable returning
            # a sparse matrix is documented and supported.
            jac=joint_jacobian,  # type: ignore[arg-type]
            verbose=verbose,
            x_scale="jac",
            loss=loss,
            f_scale=f_scale,
            ftol=ftol,
            max_nfev=max_nfev,
            method="trf",
            bounds=parameterization.bounds(),
        )

        termination_reason = _SCIPY_STATUS_REASONS.get(result.status, f"unknown_{result.status}")
        converged = result.status in (1, 2, 3, 4)

        if strict and not converged:
            from caliscope.exceptions import CalibrationError

            raise CalibrationError(
                f"Bundle adjustment did not converge: {termination_reason}\n"
                f"Pass strict=False to suppress this error and inspect the result."
            )

        new_points_xyz = parameterization.unpack_into(new_camera_array, result.x)

        bound_warnings = parameterization.bound_warnings(result.x)
        optimization_status = OptimizationStatus(
            converged=converged,
            termination_reason=termination_reason,
            iterations=result.nfev,
            final_cost=float(result.cost),
            bound_warnings=bound_warnings,
        )

        new_world_df = self.world_points.df.copy()
        new_world_df[["x_coord", "y_coord", "z_coord"]] = new_points_xyz

        return CaptureVolume(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=WorldPoints(new_world_df),
            constraints=self.constraints,
            _optimization_status=optimization_status,
        )

    def _build_constraint_arrays(
        self,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]] | None:
        """Build constraint arrays for the BA from self.constraints.

        Returns (groups_a (n_c, 4), groups_b (n_c, 4), distances (n_c,), sigmas (n_c,))
        where each endpoint group holds four row indices into world_points.df.
        A corner endpoint repeats one row index four times (its mean is exactly
        that point); a centroid endpoint names the marker's four corner rows.
        Returns None if no constraints or no valid instances.
        """
        if self.constraints is None or not (self.constraints.distances or self.constraints.centroid_distances):
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

        groups_a: list[list[int]] = []
        groups_b: list[list[int]] = []
        dists: list[float] = []
        sigmas: list[float] = []

        for dc in self.constraints.distances:
            a_static = dc.object_id_a in static_ids
            b_static = dc.object_id_b in static_ids
            if a_static != b_static:
                continue

            syncs_a = point_lookup.get((dc.object_id_a, dc.keypoint_id_a), {})
            syncs_b = point_lookup.get((dc.object_id_b, dc.keypoint_id_b), {})

            for si in self._firing_sync_indices(a_static, (syncs_a, syncs_b)):
                groups_a.append([syncs_a[si]] * 4)
                groups_b.append([syncs_b[si]] * 4)
                dists.append(dc.distance)
                sigmas.append(dc.sigma)

        for cc in self.constraints.centroid_distances:
            a_static = cc.object_id_a in static_ids
            b_static = cc.object_id_b in static_ids
            if a_static != b_static:
                continue

            corners_a = [point_lookup.get((cc.object_id_a, k), {}) for k in range(4)]
            corners_b = [point_lookup.get((cc.object_id_b, k), {}) for k in range(4)]

            # A centroid fires only where all eight corner rows exist.
            for si in self._firing_sync_indices(a_static, (*corners_a, *corners_b)):
                groups_a.append([corners_a[k][si] for k in range(4)])
                groups_b.append([corners_b[k][si] for k in range(4)])
                dists.append(cc.distance)
                sigmas.append(cc.sigma)

        if not groups_a:
            return None

        return (
            np.array(groups_a, dtype=np.int32),
            np.array(groups_b, dtype=np.int32),
            np.array(dists, dtype=np.float64),
            np.array(sigmas, dtype=np.float64),
        )

    @staticmethod
    def _firing_sync_indices(is_static: bool, lookups: tuple[dict[int, int], ...]) -> list[int]:
        """Sync indices where every endpoint lookup has a row.

        Static constraints fire once, at STATIC_SYNC_INDEX; mobile constraints
        fire per shared sync_index (STATIC_SYNC_INDEX excluded). Mixed
        static/mobile constraints are skipped by the caller before this is reached.
        """
        if is_static:
            if all(STATIC_SYNC_INDEX in lookup for lookup in lookups):
                return [STATIC_SYNC_INDEX]
            return []
        shared = set.intersection(*(set(lookup.keys()) for lookup in lookups))
        return [si for si in shared if si != STATIC_SYNC_INDEX]

    def rigidity_report(self) -> RigidityReport:
        """Measure constraint violations against current world points.

        Pure measurement, no optimization. Valid on any CaptureVolume with
        constraints, before or after optimize().
        """
        if self.constraints is None or not (self.constraints.distances or self.constraints.centroid_distances):
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
            a_static = dc.object_id_a in static_ids
            b_static = dc.object_id_b in static_ids
            if a_static != b_static:
                continue

            syncs_a = point_lookup.get((dc.object_id_a, dc.keypoint_id_a), {})
            syncs_b = point_lookup.get((dc.object_id_b, dc.keypoint_id_b), {})

            for si in self._firing_sync_indices(a_static, (syncs_a, syncs_b)):
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

        for cc in self.constraints.centroid_distances:
            a_static = cc.object_id_a in static_ids
            b_static = cc.object_id_b in static_ids
            if a_static != b_static:
                continue

            corners_a = [point_lookup.get((cc.object_id_a, k), {}) for k in range(4)]
            corners_b = [point_lookup.get((cc.object_id_b, k), {}) for k in range(4)]

            for si in self._firing_sync_indices(a_static, (*corners_a, *corners_b)):
                centroid_a = np.mean([coords[corners_a[k][si]] for k in range(4)], axis=0)
                centroid_b = np.mean([coords[corners_b[k][si]] for k in range(4)], axis=0)
                actual = float(np.linalg.norm(centroid_a - centroid_b))
                violations.append(
                    ConstraintViolation(
                        object_id_a=cc.object_id_a,
                        keypoint_id_a=-1,
                        object_id_b=cc.object_id_b,
                        keypoint_id_b=-1,
                        sync_index=si,
                        expected=cc.distance,
                        actual=actual,
                        kind="centroid",
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

        # Preserve static world points that still have observations in filtered image points.
        # Static points live at STATIC_SYNC_INDEX but their observations carry real sync_indices,
        # so the merge above drops them. Re-attach any static rows whose (object_id, keypoint_id)
        # still appears in the filtered observations.
        static_world_df = self.world_points.df[self.world_points.df["sync_index"] == STATIC_SYNC_INDEX]
        if not static_world_df.empty:
            static_obs_keys = filtered_img_df[["object_id", "keypoint_id"]].drop_duplicates()
            static_to_keep = static_world_df.merge(static_obs_keys, on=["object_id", "keypoint_id"], how="inner")
            if not static_to_keep.empty:
                filtered_world_df = pd.concat([filtered_world_df, static_to_keep], ignore_index=True)

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

        return VolumetricScaleReport(
            frame_errors=tuple(frame_errors),
            static_object_ids=self.constraints.static_object_ids if self.constraints else frozenset(),
        )

    def align_to_object(
        self,
        sync_index: int | None,
        object_id: int | None = None,
    ) -> "CaptureVolume":
        """Align the capture volume to a marker's local coordinate frame.

        The resulting world frame places the marker's center at the origin,
        X along the top edge (TL->TR), Y up the left edge (BL->TL),
        Z normal to the marker face (right-handed). A ground marker
        face-up yields a Z-up world.

        sync_index=None is valid only for static markers (world points at
        STATIC_SYNC_INDEX). Raises for non-static markers.
        """
        img_df = self.image_points.df
        world_df = self.world_points.df
        static_ids = self.constraints.static_object_ids if self.constraints else frozenset()

        # Resolve sync_index for static markers
        if sync_index is None:
            if object_id is None:
                raise ValueError("sync_index=None requires an explicit object_id")
            if object_id not in static_ids:
                raise ValueError(
                    f"sync_index=None is only valid for static markers, but object_id={object_id} is not static"
                )

        # Select image observations at this frame
        img_subset = img_df[img_df["sync_index"] == sync_index] if sync_index is not None else img_df

        if img_subset.empty:
            raise ValueError(f"No image observations at sync_index={sync_index}")

        # If object_id not specified, infer or require single-object
        if object_id is None:
            unique_objects = img_subset["object_id"].unique()
            if len(unique_objects) > 1:
                raise ValueError(
                    f"Multiple markers present at sync_index {sync_index}; "
                    f"specify object_id (available: {sorted(unique_objects)})"
                )
            object_id = int(unique_objects[0])

        # Static markers have world points at STATIC_SYNC_INDEX regardless of
        # which sync_index the caller asked for
        world_si = STATIC_SYNC_INDEX if object_id in static_ids else (sync_index if sync_index is not None else 0)

        # Filter to specified object, deduplicate obj_loc for the merge
        img_subset = img_subset[img_subset["object_id"] == object_id]
        obj_points_df = img_subset[["object_id", "keypoint_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]].drop_duplicates(
            subset=["object_id", "keypoint_id"]
        )

        world_subset = world_df[(world_df["sync_index"] == world_si) & (world_df["object_id"] == object_id)]

        if img_subset.empty:
            raise ValueError(f"No image observations for object_id={object_id} at sync_index={sync_index}")
        if world_subset.empty:
            raise ValueError(f"No world points for object_id={object_id} at sync_index={world_si}")

        # Merge on (object_id, keypoint_id) to find correspondences
        merged = pd.merge(
            world_subset[["object_id", "keypoint_id", "x_coord", "y_coord", "z_coord"]],
            obj_points_df,
            on=["object_id", "keypoint_id"],
            how="inner",
        )

        if merged["obj_loc_z"].isna().all():
            logger.info("obj_loc_z is all NaN, assuming planar board with z=0")
            merged["obj_loc_z"] = 0.0

        obj_cols = ["obj_loc_x", "obj_loc_y", "obj_loc_z"]
        valid_mask = ~merged[obj_cols].isna().any(axis=1)
        merged = merged[valid_mask]

        if len(merged) < 3:
            raise ValueError(f"Need at least 3 valid correspondences for object_id={object_id}, got {len(merged)}")

        source_points = merged[["x_coord", "y_coord", "z_coord"]].values.astype(np.float64)
        target_points = merged[obj_cols].values.astype(np.float64)

        transform = estimate_similarity_transform(source_points, target_points, rigid=True)

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

    # ------------------------------------------------------------------
    # Metric anchoring (post-BA). Each returns a new frozen CaptureVolume.
    # ------------------------------------------------------------------

    def _anchor_cam_id(self) -> int:
        """The lowest posed cam_id -- the yaw/XY convention anchor ("camera 0")."""
        posed = self.camera_array.posed_cameras
        if not posed:
            raise ValueError("No posed cameras; cannot anchor a shape-only volume.")
        return min(posed)

    def _camera_center(self, cam_id: int) -> NDArray:
        """Camera center in the current world frame: C = -R.T @ t."""
        cam = self.camera_array.cameras[cam_id]
        if cam.rotation is None or cam.translation is None:
            raise ValueError(f"Camera {cam_id} has no pose; cannot compute its center.")
        return -cam.rotation.T @ cam.translation

    def scaled(self, *cues: CameraDistance | SegmentLength | DepthObservation) -> "CaptureVolume":
        """Apply uniform metric scale recovered from one or more metric cues.

        A single cue sets scale exactly (metric / arbitrary). Multiple cues are
        combined by sigma-weighted least squares on the scale factor, using each
        cue's ``sigma_m``. Cues whose implied scales disagree by more than 2 sigma
        (propagated to scale space) trigger a ``warnings.warn``.

        ``CameraDistance`` and ``SegmentLength`` cues are user-typed and strict: a
        missing camera or keypoint raises ``ValueError``. ``DepthObservation`` cues
        are bulk estimator output (one per detection), so unresolvable ones are
        skipped with a single aggregated ``warnings.warn`` rather than aborting:
        a keypoint triangulated in fewer than two views has no world point, and
        triangulation noise can push a point behind the camera (non-positive
        depth). Zero cues, or all cues unresolvable, raise ``ValueError``.
        """
        if not cues:
            raise ValueError("scaled() requires at least one cue; got none.")

        # Compile each cue to (d_arbitrary, d_metric, sigma_m). Strict cue types
        # raise on failure; depth cues (bulk estimator output) are filtered.
        compiled: list[tuple[float, float, float]] = []
        skip_reasons: list[str] = []
        n_depth = 0
        for cue in cues:
            if isinstance(cue, DepthObservation):
                n_depth += 1
                outcome = self._compile_depth_cue(cue)
                if isinstance(outcome, str):
                    skip_reasons.append(outcome)
                else:
                    compiled.append(outcome)
            else:
                compiled.append(self._compile_cue(cue))

        if skip_reasons:
            from collections import Counter

            breakdown = ", ".join(f"{count} {reason}" for reason, count in sorted(Counter(skip_reasons).items()))
            warnings.warn(
                f"Skipped {len(skip_reasons)} of {n_depth} depth cues as unresolvable ({breakdown}).",
                stacklevel=2,
            )

        if not compiled:
            raise ValueError(f"All {len(cues)} scale cues were unresolvable; cannot determine scale.")

        d_arb = np.array([c[0] for c in compiled], dtype=np.float64)
        d_met = np.array([c[1] for c in compiled], dtype=np.float64)
        sigma = np.array([c[2] for c in compiled], dtype=np.float64)

        if len(compiled) == 1:
            scale = float(d_met[0] / d_arb[0])
        else:
            # Sigma-weighted least squares: scale = Σ(m·d/σ²) / Σ(d²/σ²).
            numerator = float(np.sum(d_met * d_arb / sigma**2))
            denominator = float(np.sum(d_arb**2 / sigma**2))
            scale = numerator / denominator

            # Disagreement diagnostic in scale space.
            implied = d_met / d_arb
            sigma_scale = sigma / d_arb
            n = len(compiled)
            for i in range(n):
                for j in range(i + 1, n):
                    combined = float(np.hypot(sigma_scale[i], sigma_scale[j]))
                    if abs(implied[i] - implied[j]) > 2.0 * combined:
                        warnings.warn(
                            f"Scale cues {i} and {j} disagree: implied scales "
                            f"{implied[i]:.6g} vs {implied[j]:.6g} differ by more than "
                            f"2 sigma ({2.0 * combined:.6g}).",
                            stacklevel=2,
                        )

        transform = SimilarityTransform(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=scale,
        )
        new_camera_array, new_world_points = apply_similarity_transform(self.camera_array, self.world_points, transform)

        return CaptureVolume(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
            constraints=self.constraints,
            _optimization_status=self._optimization_status,
        )

    def _compile_cue(self, cue: CameraDistance | SegmentLength) -> tuple[float, float, float]:
        """Reduce a strict (user-typed) cue to (d_arbitrary, d_metric, sigma_m).

        Raises ``ValueError`` on any unresolvable reference -- these cues are few
        and hand-entered, so a bad reference is a real mistake worth surfacing.
        """
        if isinstance(cue, CameraDistance):
            posed = self.camera_array.posed_cameras
            for cam_id in (cue.cam_a, cue.cam_b):
                if cam_id not in posed:
                    raise ValueError(f"CameraDistance references cam_id {cam_id}, which is not a posed camera.")
            d_arb = float(np.linalg.norm(self._camera_center(cue.cam_a) - self._camera_center(cue.cam_b)))
            if d_arb == 0.0:
                raise ValueError(f"Cameras {cue.cam_a} and {cue.cam_b} coincide; distance cue is degenerate.")
            return d_arb, float(cue.meters), float(cue.sigma_m)

        world_df = self.world_points.df

        if isinstance(cue, SegmentLength):
            coords = ["x_coord", "y_coord", "z_coord"]
            side_a = world_df[world_df["keypoint_id"] == cue.keypoint_id_a][["sync_index", "object_id", *coords]]
            side_b = world_df[world_df["keypoint_id"] == cue.keypoint_id_b][["sync_index", "object_id", *coords]]
            merged = side_a.merge(side_b, on=["sync_index", "object_id"], suffixes=("_a", "_b"))
            if merged.empty:
                raise ValueError(
                    f"SegmentLength found no frame where both keypoints "
                    f"{cue.keypoint_id_a} and {cue.keypoint_id_b} are triangulated."
                )
            deltas = merged[[f"{c}_a" for c in coords]].to_numpy() - merged[[f"{c}_b" for c in coords]].to_numpy()
            distances = np.linalg.norm(deltas, axis=1)
            d_arb = float(np.median(distances))
            return d_arb, float(cue.meters), float(cue.sigma_m)

        raise TypeError(f"Unknown scale cue type: {type(cue).__name__}")

    def _compile_depth_cue(self, cue: DepthObservation) -> tuple[float, float, float] | str:
        """Reduce a depth cue to (d_arbitrary, d_metric, sigma_m), or a skip reason.

        Depth cues are bulk estimator output, so unresolvable ones return a short
        reason string (for the caller's aggregated warning) instead of raising:
        an unposed camera, no world point (single-view keypoint), an ambiguous
        match, or a non-positive depth (point behind the camera from noise).
        """
        cam = self.camera_array.cameras.get(cue.cam_id)
        if cam is None or cam.rotation is None or cam.translation is None:
            return "unposed camera"

        world_df = self.world_points.df
        match = world_df[(world_df["sync_index"] == cue.sync_index) & (world_df["keypoint_id"] == cue.keypoint_id)]
        if match.empty:
            return "no world point"
        if len(match) > 1:
            return "ambiguous match"

        p_world = match[["x_coord", "y_coord", "z_coord"]].to_numpy()[0]
        d_arb = float((cam.rotation @ p_world + cam.translation)[2])
        if d_arb <= 0.0:
            return "non-positive depth"
        return d_arb, float(cue.depth_m), float(cue.sigma_m)

    def oriented(self, up: dict[int, NDArray]) -> "CaptureVolume":
        """Rotate so the consensus vertical becomes +Z, yaw fixed by the anchor camera.

        ``up`` maps cam_id to that camera's up vector in its own (OpenCV) frame.
        Each is mapped into the world frame (``R.T @ up_cam``), averaged, and
        normalized to a consensus vertical. The anchor camera (lowest posed cam_id)
        supplies the yaw: its optical axis, projected onto the horizontal plane,
        becomes +Y. Scale and translation are untouched.
        """
        if not up:
            raise ValueError("oriented() requires at least one up vector.")

        world_ups: list[NDArray] = []
        for cam_id, up_cam in up.items():
            cam = self.camera_array.cameras.get(cam_id)
            if cam is None or cam.rotation is None:
                raise ValueError(f"oriented() references cam_id {cam_id}, which is not a posed camera.")
            v = np.asarray(up_cam, dtype=np.float64)
            world_ups.append(cam.rotation.T @ v)

        consensus = np.mean(np.stack(world_ups), axis=0)
        norm = float(np.linalg.norm(consensus))
        if norm < 1e-9:
            raise ValueError("Consensus up vector is degenerate (per-camera verticals cancel).")
        consensus_up = consensus / norm

        anchor_cam = self.camera_array.cameras[self._anchor_cam_id()]
        assert anchor_cam.rotation is not None  # _anchor_cam_id returns a posed camera
        # Optical axis (+Z in the camera frame) expressed in the world frame.
        forward = anchor_cam.rotation.T @ np.array([0.0, 0.0, 1.0])

        rotation = world_basis_from_up_and_forward(consensus_up, forward=forward)

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

    def grounded(self, mode: Literal["lowest_point"] = "lowest_point") -> "CaptureVolume":
        """Translate so the ground sits at Z=0 and the XY origin lies under the anchor camera.

        ``mode="lowest_point"`` places the world point of least Z at Z=0 (call after
        ``oriented()`` so Z is vertical). XY origin is set under the anchor camera
        (lowest posed cam_id). No rotation or scale change.
        """
        if mode != "lowest_point":
            raise ValueError(f"grounded() only supports mode='lowest_point', got {mode!r}.")

        min_z = float(self.world_points.df["z_coord"].min())
        anchor_center = self._camera_center(self._anchor_cam_id())
        offset = np.array([anchor_center[0], anchor_center[1], min_z], dtype=np.float64)

        transform = SimilarityTransform(
            rotation=np.eye(3, dtype=np.float64),
            translation=-offset,
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

    def centered(self) -> "CaptureVolume":
        """Translate so the XY origin is the centroid of posed camera centers.

        Z is untouched. Call after ``grounded()`` to keep the floor at Z=0
        while distributing cameras evenly around the origin.
        """
        centers = np.array([self._camera_center(cid) for cid in self.camera_array.posed_cameras])
        centroid_xy = centers[:, :2].mean(axis=0)
        offset = np.array([centroid_xy[0], centroid_xy[1], 0.0], dtype=np.float64)

        transform = SimilarityTransform(
            rotation=np.eye(3, dtype=np.float64),
            translation=-offset,
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
