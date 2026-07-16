"""Tests for core/calibrate_extrinsics.py — the use-case function."""

from __future__ import annotations

import numpy as np
import pytest

import pandas as pd

from caliscope.core.calibrate_extrinsics import (
    CalibrationRun,
    _count_firing_cross_face_rows,
    _validate_two_sided_extraction,
    calibrate_extrinsics,
    refresh_run,
)
from caliscope.core.constraints import DistanceConstraint
from caliscope.core.point_data import ImagePoints
from caliscope.exceptions import CalibrationError
from caliscope.synthetic import strip_intrinsics
from caliscope.synthetic.scene_factories import wand_scene_with_constraints
from caliscope.task_manager.cancellation import CancellationToken


class TestEndToEndBlind:
    def test_blind_intrinsics_converge(self):
        scene, constraints = wand_scene_with_constraints(include_static=False)
        cameras = strip_intrinsics(scene.intrinsics_only_cameras())

        result = calibrate_extrinsics(
            scene.image_points_noisy,
            cameras,
            constraints,
        )

        assert isinstance(result, CalibrationRun)
        assert result.capture_volume.optimization_status is not None
        assert result.capture_volume.optimization_status.converged
        assert result.synthesized_cam_ids == frozenset(cameras.cameras.keys())

        for est in result.intrinsic_estimates:
            true_cam = scene.camera_array.cameras[est.cam_id]
            assert true_cam.matrix is not None
            f_true = float(true_cam.matrix[0, 0])
            f_err_pct = abs(est.f_recovered - f_true) / f_true * 100
            assert f_err_pct < 2.0, f"Cam {est.cam_id}: f error {f_err_pct:.2f}%"

        report = result.capture_volume.rigidity_report()
        errors_mm = [abs(v.actual - v.expected) * 1000 for v in report.violations]
        rig_rmse = float(np.sqrt(np.mean(np.array(errors_mm) ** 2)))
        assert rig_rmse < 2.0, f"Rigidity RMSE {rig_rmse:.3f}mm"


class TestProvidedIntrinsics:
    def test_synthesized_cam_ids_empty(self):
        scene, constraints = wand_scene_with_constraints(include_static=False)
        cameras = scene.intrinsics_only_cameras()

        result = calibrate_extrinsics(
            scene.image_points_noisy,
            cameras,
            constraints,
        )

        assert result.synthesized_cam_ids == frozenset()
        for est in result.intrinsic_estimates:
            true_cam = scene.camera_array.cameras[est.cam_id]
            assert true_cam.matrix is not None
            assert abs(est.f_initial - float(true_cam.matrix[0, 0])) < 1e-6


class TestStaticMarkerGuard:
    def test_corrupted_static_marker_dropped(self):
        scene, constraints = wand_scene_with_constraints(include_static=True)
        cameras = scene.intrinsics_only_cameras()

        # Corrupt one corner of a static marker to distort its triangulated shape
        target_obj_id = sorted(constraints.static_object_ids)[0]
        img_df = scene.image_points_noisy.df.copy()
        mask = (img_df["object_id"] == target_obj_id) & (img_df["keypoint_id"] == 0)
        img_df.loc[mask, "img_loc_x"] += 500.0
        img_df.loc[mask, "img_loc_y"] += 300.0

        from caliscope.core.point_data import ImagePoints

        corrupted = ImagePoints(img_df)

        result = calibrate_extrinsics(
            corrupted,
            cameras,
            constraints,
        )

        assert target_obj_id in result.dropped_static_markers
        assert result.capture_volume.optimization_status is not None
        assert result.capture_volume.optimization_status.converged


class TestProgressAndCancellation:
    def test_progress_monotonic_and_ends_at_100(self):
        scene, constraints = wand_scene_with_constraints(include_static=False)
        cameras = scene.intrinsics_only_cameras()

        reports: list[tuple[int, str]] = []
        calibrate_extrinsics(
            scene.image_points_noisy,
            cameras,
            constraints,
            progress=lambda pct, msg: reports.append((pct, msg)),
        )

        pcts = [r[0] for r in reports]
        assert pcts[-1] == 100
        assert pcts == sorted(pcts)

    def test_cancelled_token_raises(self):
        scene, constraints = wand_scene_with_constraints(include_static=False)
        cameras = scene.intrinsics_only_cameras()

        token = CancellationToken()
        token.cancel()

        with pytest.raises(InterruptedError):
            calibrate_extrinsics(
                scene.image_points_noisy,
                cameras,
                constraints,
                cancellation_token=token,
            )


class TestRefreshRun:
    def test_preserves_anchors_updates_recovered(self):
        scene, constraints = wand_scene_with_constraints(include_static=False)
        cameras = scene.intrinsics_only_cameras()

        original = calibrate_extrinsics(
            scene.image_points_noisy,
            cameras,
            constraints,
        )

        cv = original.capture_volume
        filtered = cv.filter_by_percentile_error(2.5)
        reoptimized = filtered.optimize(refine_intrinsics=True)

        refreshed = refresh_run(original, reoptimized)

        for est_orig, est_new in zip(original.intrinsic_estimates, refreshed.intrinsic_estimates):
            assert est_orig.f_initial == est_new.f_initial
            assert est_orig.k1_initial == est_new.k1_initial

        assert refreshed.synthesized_cam_ids == original.synthesized_cam_ids
        assert refreshed.intrinsic_refinement_gated == original.intrinsic_refinement_gated


def _image_points_with_objects(object_ids: list[int], back_z: float = 0.006) -> ImagePoints:
    """Minimal valid ImagePoints observing the given object_ids.

    object_id 1 rows carry obj_loc_z = back_z (the extraction-time thickness
    stamp); object 0 rows sit at z=0.
    """
    rows = []
    for oid in object_ids:
        for kid in (0, 1):
            rows.append(
                {
                    "sync_index": 0,
                    "cam_id": 0,
                    "object_id": oid,
                    "keypoint_id": kid,
                    "img_loc_x": 100.0,
                    "img_loc_y": 100.0,
                    "obj_loc_x": 0.05 * kid,
                    "obj_loc_y": 0.0,
                    "obj_loc_z": back_z if oid == 1 else 0.0,
                }
            )
    return ImagePoints(pd.DataFrame(rows))


class TestTwoSidedExtractionGuard:
    def test_consistent_thin_and_thick_pass(self):
        _validate_two_sided_extraction(_image_points_with_objects([0]), thickness_m=0.0)
        _validate_two_sided_extraction(_image_points_with_objects([0, 1]), thickness_m=0.006)

    def test_thickness_set_but_extraction_has_no_back_face(self):
        with pytest.raises(CalibrationError, match="no back-face observations"):
            _validate_two_sided_extraction(_image_points_with_objects([0]), thickness_m=0.006)

    def test_thickness_zero_but_extraction_has_back_face(self):
        with pytest.raises(CalibrationError, match="thickness is 0"):
            _validate_two_sided_extraction(_image_points_with_objects([0, 1]), thickness_m=0.0)

    def test_thickness_changed_between_extraction_and_calibration(self):
        with pytest.raises(CalibrationError, match="thickness changed"):
            _validate_two_sided_extraction(_image_points_with_objects([0, 1], back_z=0.006), thickness_m=0.012)


class TestCrossFaceFiringCount:
    def _cross_row(self, kid_a: int, kid_b: int) -> DistanceConstraint:
        return DistanceConstraint(
            object_id_a=0, keypoint_id_a=kid_a, object_id_b=1, keypoint_id_b=kid_b, distance=0.006, sigma=0.0005
        )

    def test_counts_only_rows_with_shared_sync_index(self):
        world_df = pd.DataFrame(
            [
                # corner 0 triangulated on both faces at sync 5 -> tie fires
                {"sync_index": 5, "object_id": 0, "keypoint_id": 0},
                {"sync_index": 5, "object_id": 1, "keypoint_id": 0},
                # corner 1: faces triangulated at DIFFERENT syncs -> no fire
                {"sync_index": 6, "object_id": 0, "keypoint_id": 1},
                {"sync_index": 7, "object_id": 1, "keypoint_id": 1},
            ]
        )
        rows = (self._cross_row(0, 0), self._cross_row(1, 1))
        assert _count_firing_cross_face_rows(world_df, rows) == 1

    def test_intra_face_rows_ignored(self):
        world_df = pd.DataFrame([{"sync_index": 5, "object_id": 0, "keypoint_id": 0}])
        intra = DistanceConstraint(
            object_id_a=0, keypoint_id_a=0, object_id_b=0, keypoint_id_b=1, distance=0.05, sigma=0.002
        )
        assert _count_firing_cross_face_rows(world_df, (intra,)) == 0


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("End-to-end blind...")
    TestEndToEndBlind().test_blind_intrinsics_converge()
    print("  PASSED")

    print("Provided intrinsics...")
    TestProvidedIntrinsics().test_synthesized_cam_ids_empty()
    print("  PASSED")

    print("Static marker guard...")
    TestStaticMarkerGuard().test_corrupted_static_marker_dropped()
    print("  PASSED")

    print("Progress monotonic...")
    TestProgressAndCancellation().test_progress_monotonic_and_ends_at_100()
    print("  PASSED")

    print("Cancelled token...")
    TestProgressAndCancellation().test_cancelled_token_raises()
    print("  PASSED")

    print("Refresh run...")
    TestRefreshRun().test_preserves_anchors_updates_recovered()
    print("  PASSED")
