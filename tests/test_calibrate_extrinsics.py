"""Tests for core/calibrate_extrinsics.py — the use-case function."""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.core.calibrate_extrinsics import (
    ExtrinsicCalibrationResult,
    calibrate_extrinsics,
    refresh_result,
)
from caliscope.synthetic.scene_factories import wand_scene_with_constraints
from caliscope.task_manager.cancellation import CancellationToken


def _strip_intrinsics(scene):
    """Return intrinsics-only cameras with matrix/distortions zeroed out."""
    cameras = scene.intrinsics_only_cameras()
    for cam in cameras.cameras.values():
        cam.matrix = None
        cam.distortions = None
    return cameras


class TestEndToEndBlind:
    def test_blind_intrinsics_converge(self):
        scene, constraints = wand_scene_with_constraints(include_static=False)
        cameras = _strip_intrinsics(scene)

        result = calibrate_extrinsics(
            scene.image_points_noisy,
            cameras,
            constraints,
        )

        assert isinstance(result, ExtrinsicCalibrationResult)
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


class TestRefreshResult:
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

        refreshed = refresh_result(original, reoptimized)

        for est_orig, est_new in zip(original.intrinsic_estimates, refreshed.intrinsic_estimates):
            assert est_orig.f_initial == est_new.f_initial
            assert est_orig.k1_initial == est_new.k1_initial

        assert refreshed.synthesized_cam_ids == original.synthesized_cam_ids


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

    print("Refresh result...")
    TestRefreshResult().test_preserves_anchors_updates_recovered()
    print("  PASSED")
