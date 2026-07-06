"""Single seam for driving synthetic scenes through the production pipeline.

Every production-pipeline test goes through run_production_pipeline, which calls
the real calibrate_extrinsics() use-case function with production defaults, then
Procrustes-aligns the result to ground truth for pose comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

from caliscope.core.calibrate_extrinsics import ExtrinsicCalibrationResult, calibrate_extrinsics
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import ImagePoints
from caliscope.synthetic import SyntheticScene, strip_intrinsics

from tests.synthetic.assertions import PoseError, align_to_ground_truth, pose_error


@dataclass(frozen=True)
class ProductionRun:
    result: ExtrinsicCalibrationResult
    aligned_volume: CaptureVolume  # Procrustes-aligned to ground truth
    pose_errors: dict[int, PoseError]  # cam_id -> error vs ground truth

    @property
    def max_rotation_deg(self) -> float:
        return max(e.rotation_deg for e in self.pose_errors.values())

    @property
    def max_translation_m(self) -> float:
        return max(e.translation_m for e in self.pose_errors.values())


def run_production_pipeline(
    scene: SyntheticScene,
    *,
    image_points: ImagePoints | None = None,  # override, e.g. corrupted points
    constraints: ConstraintSet | None = None,
    blind: bool = False,  # also strip intrinsics
    refine_intrinsics: bool = True,
    filter_percentile: float = 2.5,
) -> ProductionRun:
    """Drive a scene through calibrate_extrinsics() and align to ground truth.

    Defaults mirror production defaults exactly; no solver knobs the presenter
    does not have. Pass image_points to substitute corrupted observations, or
    blind=True to also strip intrinsics and exercise default synthesis.
    """
    cameras = scene.intrinsics_only_cameras()
    if blind:
        cameras = strip_intrinsics(cameras)
    result = calibrate_extrinsics(
        image_points if image_points is not None else scene.image_points_noisy,
        cameras,
        constraints,
        refine_intrinsics=refine_intrinsics,
        filter_percentile=filter_percentile,
    )
    aligned = align_to_ground_truth(result.capture_volume, scene)
    errors = {
        cam_id: pose_error(aligned.camera_array.cameras[cam_id], scene.camera_array.cameras[cam_id])
        for cam_id in aligned.camera_array.posed_cameras
    }
    return ProductionRun(result=result, aligned_volume=aligned, pose_errors=errors)
