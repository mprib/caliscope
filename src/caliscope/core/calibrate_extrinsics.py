"""Use-case function for extrinsic calibration.

Pipeline: synthesize blind intrinsics → bootstrap → static-marker guard →
optimize → filter → optimize. Returns an ExtrinsicCalibrationResult with
recovered intrinsic estimates, bound warnings, depth ratios, and dropped markers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass


from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bundle_parameterization import BoundWarning, IntrinsicEstimate
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.constraints import ConstraintSet, RigidityReport
from caliscope.core.point_data import ImagePoints
from caliscope.core.scale_accuracy import compute_depth_ratios
from caliscope.task_manager.cancellation import CancellationToken

logger = logging.getLogger(__name__)

# Below this per-camera near/far depth ratio, focal length is not jointly
# observable with extrinsics, so refining it drifts f and couples scale error
# into camera translation. Same bound the synthetic premise check uses
# (_check_intrinsic_perturbation_premises, scene_factories.py) and the E4
# negative control; measured separation is ring 1.29 vs wand 2.25.
MIN_DEPTH_RATIO_FOR_INTRINSIC_REFINEMENT = 2.0


@dataclass(frozen=True)
class ExtrinsicCalibrationResult:
    capture_volume: CaptureVolume
    intrinsic_estimates: tuple[IntrinsicEstimate, ...]
    synthesized_cam_ids: frozenset[int]
    bound_warnings: tuple[BoundWarning, ...]
    dropped_static_markers: tuple[int, ...]
    # Per-camera depth ratios of the final volume; may differ from the values the
    # refinement gate acted on (which were computed on the post-first-linear-pass
    # volume, before filtering and the final optimize).
    depth_ratios: dict[int, float]
    intrinsic_refinement_gated: bool


def calibrate_extrinsics(
    image_points: ImagePoints,
    camera_array: CameraArray,
    constraints: ConstraintSet | None,
    *,
    refine_intrinsics: bool = True,
    filter_percentile: float = 2.5,
    cancellation_token: CancellationToken | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> ExtrinsicCalibrationResult:
    """Run the full extrinsic calibration pipeline.

    Synthesizes blind intrinsics for uncalibrated cameras, bootstraps poses,
    applies static-marker guard, runs two optimization passes with outlier
    filtering between them.
    """

    def _progress(pct: int, msg: str) -> None:
        if progress is not None:
            progress(pct, msg)

    def _check_cancelled() -> None:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            raise InterruptedError("Calibration cancelled")

    # 1. Prepare cameras
    _progress(5, "Preparing cameras")
    cameras = deepcopy(camera_array)
    synthesized: set[int] = set()
    for cam in cameras.cameras.values():
        if cam.ignore:
            continue
        if cam.matrix is None or cam.distortions is None:
            synthesized.add(cam.cam_id)
            cam.synthesize_default_intrinsics()

    # 2. Capture initial intrinsic anchors
    anchors: dict[int, tuple[float, float, float]] = {}
    for cam in cameras.cameras.values():
        if cam.ignore or cam.matrix is None or cam.distortions is None:
            continue
        anchors[cam.cam_id] = (
            float(cam.matrix[0, 0]),
            float(cam.distortions[0]),
            float(cam.distortions[1]),
        )

    _check_cancelled()

    # 3. Bootstrap
    _progress(15, "Bootstrapping poses")
    capture_volume = CaptureVolume.bootstrap(image_points, cameras, constraints=constraints)

    _check_cancelled()

    # 4. Static-marker guard
    dropped_markers: list[int] = []
    if constraints is not None and constraints.static_object_ids:
        report = capture_volume.rigidity_report()
        # Gate on intra-marker rigidity only. per_object_rmse_mm attributes
        # cross-object violations (e.g. a static-static center link's) to both
        # endpoints, which would conflate tape-measure disagreement with "did
        # this marker move". Filter to intra-marker violations before aggregating.
        intra_violations = tuple(v for v in report.violations if v.object_id_a == v.object_id_b)
        obj_rmse = RigidityReport(violations=intra_violations).per_object_rmse_mm

        for obj_id in sorted(constraints.static_object_ids):
            rmse = obj_rmse.get(obj_id, 0.0)
            max_intra_mm = _max_intra_distance_mm(constraints, obj_id)
            if max_intra_mm > 0 and rmse > 0.25 * max_intra_mm:
                logger.warning(
                    f"Dropping static marker {obj_id}: rigidity RMSE {rmse:.1f}mm "
                    f"> 25% of max intra-distance {max_intra_mm:.1f}mm"
                )
                dropped_markers.append(obj_id)

        if dropped_markers:
            dropped_set = set(dropped_markers)
            filtered_img_df = image_points.df[~image_points.df["object_id"].isin(dropped_set)]
            image_points = ImagePoints(filtered_img_df.reset_index(drop=True))

            filtered_distances = tuple(
                d
                for d in constraints.distances
                if d.object_id_a not in dropped_set and d.object_id_b not in dropped_set
            )
            filtered_centroids = tuple(
                c
                for c in constraints.centroid_distances
                if c.object_id_a not in dropped_set and c.object_id_b not in dropped_set
            )
            filtered_statics = constraints.static_object_ids - frozenset(dropped_set)
            constraints = ConstraintSet(
                distances=filtered_distances,
                static_object_ids=filtered_statics,
                centroid_distances=filtered_centroids,
            )

            _progress(20, "Re-bootstrapping after dropping markers")
            cameras = deepcopy(camera_array)
            for cam in cameras.cameras.values():
                if cam.ignore:
                    continue
                if cam.cam_id in synthesized:
                    cam.synthesize_default_intrinsics()
            capture_volume = CaptureVolume.bootstrap(image_points, cameras, constraints=constraints)

    _check_cancelled()

    # 5. Linear optimize (fast convergence to the basin). Always extrinsics-only:
    # this pass exists to reach the basin, and refining here is either harmful
    # (weak geometry) or unnecessary (strong geometry — later passes handle it).
    _progress(40, "Optimizing")
    capture_volume = capture_volume.optimize(refine_intrinsics=False)

    _check_cancelled()

    # Depth-ratio gate: refine intrinsics only where focal is jointly observable
    # in every camera. compute_depth_ratios returns NaN for a camera with too few
    # positive-depth points; NaN >= threshold is False, so an all() check gates a
    # degenerate camera off naturally. Do not fold this back to min() — min() over
    # NaN is insertion-order dependent and can let a NaN camera slip through.
    depth_ratios = compute_depth_ratios(capture_volume)
    effective_refine = (
        refine_intrinsics
        and bool(depth_ratios)
        and all(r >= MIN_DEPTH_RATIO_FOR_INTRINSIC_REFINEMENT for r in depth_ratios.values())
    )
    intrinsic_refinement_gated = refine_intrinsics and not effective_refine
    if intrinsic_refinement_gated:
        logger.warning(
            f"Intrinsic refinement requested but gated off (need every camera >= "
            f"{MIN_DEPTH_RATIO_FOR_INTRINSIC_REFINEMENT}). Per-camera depth ratios: {depth_ratios}"
        )

    # 6. Robust refinement (warm-started, protects poses from outliers)
    _progress(55, "Robust refinement")
    f_scale = capture_volume.pixel_f_scale(px=1.0)
    capture_volume = capture_volume.optimize(
        refine_intrinsics=effective_refine,
        loss="soft_l1",
        f_scale=f_scale,
        max_nfev=2000,
        ftol=1e-4,
        strict=False,
    )

    _check_cancelled()

    # 7. Filter outliers
    _progress(75, "Filtering outliers")
    capture_volume = capture_volume.filter_by_percentile_error(filter_percentile)

    _check_cancelled()

    # 8. Final optimize (clean data, linear is sufficient)
    _progress(90, "Re-optimizing")
    capture_volume = capture_volume.optimize(refine_intrinsics=effective_refine)

    # 9. Assemble result
    _progress(100, "Done")
    return _build_result(
        capture_volume=capture_volume,
        anchors=anchors,
        synthesized_cam_ids=frozenset(synthesized),
        dropped_static_markers=tuple(dropped_markers),
        intrinsic_refinement_gated=intrinsic_refinement_gated,
    )


def refresh_result(
    previous: ExtrinsicCalibrationResult,
    capture_volume: CaptureVolume,
) -> ExtrinsicCalibrationResult:
    """Rebuild the result around a re-optimized capture volume.

    Preserved: initial anchors, synthesized_cam_ids, dropped markers.
    Recomputed: intrinsic estimates, bound warnings, depth ratios.
    """
    anchors: dict[int, tuple[float, float, float]] = {}
    for est in previous.intrinsic_estimates:
        anchors[est.cam_id] = (est.f_initial, est.k1_initial, est.k2_initial)

    return _build_result(
        capture_volume=capture_volume,
        anchors=anchors,
        synthesized_cam_ids=previous.synthesized_cam_ids,
        dropped_static_markers=previous.dropped_static_markers,
        intrinsic_refinement_gated=previous.intrinsic_refinement_gated,
    )


def _build_result(
    capture_volume: CaptureVolume,
    anchors: dict[int, tuple[float, float, float]],
    synthesized_cam_ids: frozenset[int],
    dropped_static_markers: tuple[int, ...],
    intrinsic_refinement_gated: bool,
) -> ExtrinsicCalibrationResult:
    estimates: list[IntrinsicEstimate] = []
    for cam_id, cam in capture_volume.camera_array.posed_cameras.items():
        if cam_id not in anchors or cam.matrix is None or cam.distortions is None:
            continue
        f_init, k1_init, k2_init = anchors[cam_id]
        estimates.append(
            IntrinsicEstimate(
                cam_id=cam_id,
                f_recovered=float(cam.matrix[0, 0]),
                k1_recovered=float(cam.distortions[0]),
                k2_recovered=float(cam.distortions[1]),
                f_initial=f_init,
                k1_initial=k1_init,
                k2_initial=k2_init,
            )
        )

    status = capture_volume.optimization_status
    bound_warnings = status.bound_warnings if status is not None else ()

    return ExtrinsicCalibrationResult(
        capture_volume=capture_volume,
        intrinsic_estimates=tuple(estimates),
        synthesized_cam_ids=synthesized_cam_ids,
        bound_warnings=bound_warnings,
        dropped_static_markers=dropped_static_markers,
        depth_ratios=compute_depth_ratios(capture_volume),
        intrinsic_refinement_gated=intrinsic_refinement_gated,
    )


def _max_intra_distance_mm(constraints: ConstraintSet, object_id: int) -> float:
    """Maximum distance between any two keypoints on a single marker, in mm."""
    max_d = 0.0
    for dc in constraints.distances:
        if dc.object_id_a == object_id and dc.object_id_b == object_id:
            max_d = max(max_d, dc.distance)
    return max_d * 1000.0
