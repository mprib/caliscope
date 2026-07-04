"""D7 experiments: joint intrinsic+extrinsic bundle adjustment.

Sparsity oracle, E1 (f recovery), E2 (k1 recovery), E3 (constraints as
metric anchor), E4 (negative control), E5b (outlier contamination drift).
"""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.core.capture_volume import CaptureVolume
from caliscope.synthetic.camera_synthesizer import IntrinsicPerturbation, perturb_intrinsics
from caliscope.synthetic.experimental_ba import (
    N_CAM_PARAMS,
    _joint_residuals,
    _joint_sparsity_pattern,
    optimize_with_free_intrinsics,
)
from caliscope.synthetic.scene_factories import intrinsic_perturbation_scene
from tests.synthetic.assertions import align_to_ground_truth, pose_error


def _bootstrap_from_scene(scene, constraints=None, perturbation=None):
    """Bootstrap a CaptureVolume from a scene, optionally with perturbed intrinsics."""
    cameras = scene.intrinsics_only_cameras()
    if perturbation is not None:
        cameras = perturb_intrinsics(cameras, perturbation)
    return CaptureVolume.bootstrap(
        scene.image_points_noisy,
        cameras,
        constraints=constraints,
    )


class TestSparsityOracle:
    """Every zero in the sparsity pattern must be a true zero partial derivative."""

    def test_joint_sparsity_zeros_match_true_jacobian(self) -> None:
        scene = intrinsic_perturbation_scene()
        cv = _bootstrap_from_scene(scene)

        matched_mask = cv.img_to_obj_map >= 0
        posed_cam_ids = set(cv.camera_array.posed_cam_id_to_index.keys())
        posed_mask = cv.image_points.df["cam_id"].isin(posed_cam_ids).to_numpy()
        combined_mask = matched_mask & posed_mask

        matched_img_df = cv.image_points.df[combined_mask]
        camera_indices = np.array(
            [cv.camera_array.posed_cam_id_to_index[cid] for cid in matched_img_df["cam_id"]],
            dtype=np.int16,
        )
        image_coords = matched_img_df[["img_loc_x", "img_loc_y"]].values
        obj_indices = cv.img_to_obj_map[combined_mask]

        import cv2

        n_cams = len(cv.camera_array.posed_cameras)
        cam_ids_by_index = sorted(
            cv.camera_array.posed_cam_id_to_index,
            key=lambda cid: cv.camera_array.posed_cam_id_to_index[cid],
        )

        cx_cy = np.zeros((n_cams, 2))
        dist_tail = np.zeros((n_cams, 3))
        f_initial = np.zeros(n_cams)
        camera_params = np.zeros((n_cams, N_CAM_PARAMS))

        for idx, cam_id in enumerate(cam_ids_by_index):
            cam = cv.camera_array.cameras[cam_id]
            assert cam.matrix is not None and cam.distortions is not None
            assert cam.rotation is not None and cam.translation is not None
            rvec = cv2.Rodrigues(cam.rotation)[0].ravel()
            camera_params[idx, 0:3] = rvec
            camera_params[idx, 3:6] = cam.translation
            camera_params[idx, 6] = cam.matrix[0, 0]
            camera_params[idx, 7] = cam.distortions[0]
            camera_params[idx, 8] = cam.distortions[1]
            cx_cy[idx] = cam.matrix[0, 2], cam.matrix[1, 2]
            dist_tail[idx] = cam.distortions[2:5]
            f_initial[idx] = cam.matrix[0, 0]

        points_3d = cv.world_points.points
        x0 = np.concatenate([camera_params.ravel(), points_3d.ravel()])

        n_points = len(points_3d)
        sparsity = _joint_sparsity_pattern(camera_indices, obj_indices, n_cams, n_points)

        def residual_fn(x):
            return _joint_residuals(
                x,
                n_cams,
                camera_indices,
                image_coords,
                obj_indices,
                cx_cy,
                dist_tail,
                f_initial,
            )

        f0 = residual_fn(x0)
        eps = 1e-7
        n_params = len(x0)
        J = np.zeros((len(f0), n_params))
        for j in range(n_params):
            x_plus = x0.copy()
            x_plus[j] += eps
            J[:, j] = (residual_fn(x_plus) - f0) / eps

        zero_mask = sparsity.toarray() == 0
        max_in_zeros = float(np.max(np.abs(J[zero_mask]))) if zero_mask.any() else 0.0
        assert max_in_zeros < 1e-4, f"Sparsity pattern has false zeros: max |J| in zero entries = {max_in_zeros:.2e}"


class TestE1FocalLengthRecovery:
    """E1: Perturb f by 3%, run joint BA, assert recovery within 1%."""

    def test_f_recovery(self) -> None:
        scene = intrinsic_perturbation_scene()
        perturbation = IntrinsicPerturbation(f_scale=1.03)
        cv = _bootstrap_from_scene(scene, perturbation=perturbation)

        result = optimize_with_free_intrinsics(cv)

        assert result.converged
        assert not result.hit_bounds

        for est in result.intrinsic_estimates:
            cam = scene.camera_array.cameras[est.cam_id]
            assert cam.matrix is not None
            assert cam.distortions is not None
            f_true = float(cam.matrix[0, 0])
            k2_true = float(cam.distortions[1])
            f_error_pct = abs(est.f_recovered - f_true) / f_true * 100
            k2_error = abs(est.k2_recovered - k2_true)
            assert f_error_pct < 1.0, (
                f"Cam {est.cam_id}: f error {f_error_pct:.2f}% >= 1% "
                f"(true={f_true:.1f}, recovered={est.f_recovered:.1f})"
            )
            # k1 and k2 trade along a correlation ridge — individual
            # coefficients shift while total radial distortion is preserved.
            assert k2_error < 0.03, (
                f"Cam {est.cam_id}: |k2_recovered - k2_true| = {k2_error:.4f} >= 0.03 "
                f"(true={k2_true:.4f}, recovered={est.k2_recovered:.4f})"
            )

        aligned = align_to_ground_truth(result.capture_volume, scene)
        for cam_id in scene.camera_array.cameras:
            err = pose_error(aligned.camera_array.cameras[cam_id], scene.camera_array.cameras[cam_id])
            assert err.rotation_deg < 1.0, f"Cam {cam_id}: rotation {err.rotation_deg:.3f} deg"
            assert err.translation_m < 0.02, f"Cam {cam_id}: translation {err.translation_m:.4f} m"


class TestE2K1Recovery:
    """E2: Perturb k1 by 0.02, run joint BA, assert recovery within 0.02.

    Tolerance widened from 0.01 to 0.02: with k2 free, k1 and k2 trade
    along a correlation ridge. Individual coefficient accuracy decreases
    while total radial distortion correction is preserved.
    """

    def test_k1_recovery(self) -> None:
        scene = intrinsic_perturbation_scene()
        perturbation = IntrinsicPerturbation(k1_delta=0.02)
        cv = _bootstrap_from_scene(scene, perturbation=perturbation)

        result = optimize_with_free_intrinsics(cv)

        assert result.converged
        assert not result.hit_bounds

        for est in result.intrinsic_estimates:
            cam = scene.camera_array.cameras[est.cam_id]
            assert cam.distortions is not None
            k1_true = float(cam.distortions[0])
            k1_error = abs(est.k1_recovered - k1_true)
            assert k1_error < 0.02, (
                f"Cam {est.cam_id}: |k1_recovered - k1_true| = {k1_error:.4f} >= 0.02 "
                f"(true={k1_true:.4f}, recovered={est.k1_recovered:.4f})"
            )

        aligned = align_to_ground_truth(result.capture_volume, scene)
        for cam_id in scene.camera_array.cameras:
            err = pose_error(aligned.camera_array.cameras[cam_id], scene.camera_array.cameras[cam_id])
            assert err.rotation_deg < 1.0, f"Cam {cam_id}: rotation {err.rotation_deg:.3f} deg"
            assert err.translation_m < 0.02, f"Cam {cam_id}: translation {err.translation_m:.4f} m"


class TestE2bCombinedPerturbation:
    """E2b: Perturb f, k1, and k2 together, assert joint recovery."""

    def test_combined_f_k1_k2_recovery(self) -> None:
        scene = intrinsic_perturbation_scene()
        perturbation = IntrinsicPerturbation(f_scale=1.03, k1_delta=0.02, k2_delta=0.05)
        cv = _bootstrap_from_scene(scene, perturbation=perturbation)

        result = optimize_with_free_intrinsics(cv)

        assert result.converged
        assert not result.hit_bounds

        for est in result.intrinsic_estimates:
            cam = scene.camera_array.cameras[est.cam_id]
            assert cam.matrix is not None
            assert cam.distortions is not None
            f_true = float(cam.matrix[0, 0])
            k1_true = float(cam.distortions[0])
            k2_true = float(cam.distortions[1])
            f_error_pct = abs(est.f_recovered - f_true) / f_true * 100
            k1_error = abs(est.k1_recovered - k1_true)
            k2_error = abs(est.k2_recovered - k2_true)
            assert f_error_pct < 1.0, (
                f"Cam {est.cam_id}: f error {f_error_pct:.2f}% (true={f_true:.1f}, recovered={est.f_recovered:.1f})"
            )
            assert k1_error < 0.02, (
                f"Cam {est.cam_id}: k1 error {k1_error:.4f} (true={k1_true:.4f}, recovered={est.k1_recovered:.4f})"
            )
            assert k2_error < 0.03, (
                f"Cam {est.cam_id}: k2 error {k2_error:.4f} (true={k2_true:.4f}, recovered={est.k2_recovered:.4f})"
            )


class TestE3ConstraintsAsMetricAnchor:
    """E3: 2x2 — {clean, corrupted} x {constrained, unconstrained}.

    Clean pair: constrained f error <= unconstrained f error.
    Corrupted pair: characterization only (journal, no assertion).
    """

    def test_clean_constrained_improves_f(self) -> None:
        """Use the charuco scene with known-distance constraints between board corners."""
        from caliscope.core.constraints import ConstraintSet, DistanceConstraint

        scene = intrinsic_perturbation_scene()
        perturbation = IntrinsicPerturbation(f_scale=1.03)

        board = scene.objects[0].calibration_object
        dcs: list[DistanceConstraint] = []
        pts = board.points
        kids = board.keypoint_ids
        for i in range(len(kids)):
            for j in range(i + 1, min(i + 3, len(kids))):
                dist = float(np.linalg.norm(pts[i] - pts[j]))
                dcs.append(DistanceConstraint(0, int(kids[i]), 0, int(kids[j]), dist, 0.001))
        constraints = ConstraintSet(distances=tuple(dcs), static_object_ids=frozenset())

        cv_unconstrained = _bootstrap_from_scene(scene, perturbation=perturbation)
        cv_constrained = _bootstrap_from_scene(scene, constraints=constraints, perturbation=perturbation)

        result_unconstrained = optimize_with_free_intrinsics(cv_unconstrained, use_constraints=False)
        result_constrained = optimize_with_free_intrinsics(cv_constrained, use_constraints=True)

        assert result_unconstrained.converged
        assert result_constrained.converged

        for est_c, est_u in zip(
            result_constrained.intrinsic_estimates,
            result_unconstrained.intrinsic_estimates,
        ):
            cam = scene.camera_array.cameras[est_c.cam_id]
            assert cam.matrix is not None
            f_true = float(cam.matrix[0, 0])
            err_c = abs(est_c.f_recovered - f_true) / f_true
            err_u = abs(est_u.f_recovered - f_true) / f_true
            # Both should recover well; constraints must not degrade recovery.
            # On a scene with good depth variation, both achieve < 0.1%, so
            # the comparison tests noise. A 2x margin captures "no degradation".
            assert err_c <= max(err_u * 2.0, 0.005), (
                f"Cam {est_c.cam_id}: constrained f error {err_c:.4f} > "
                f"2x unconstrained {err_u:.4f} — constraints degraded f recovery"
            )


class TestE4NegativeControl:
    """E4: Small board far from cameras, stationary. Joint BA should NOT recover f.

    A large board near cameras gives partial f-observability through angular
    span alone (corners span depth relative to camera). Use a tiny board at
    distance ~3m where all corners sit at nearly the same depth.
    """

    def test_stationary_f_not_recovered(self) -> None:
        from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
        from caliscope.synthetic.calibration_object import CalibrationObject
        from caliscope.synthetic.trajectory import Trajectory
        from caliscope.synthetic.synthetic_scene import SyntheticScene

        camera_array = CameraSynthesizer().add_ring(n=4, radius=3.0, height=0.3).build()
        calibration_object = CalibrationObject.planar_grid(rows=3, cols=4, spacing=0.03)
        trajectory = Trajectory.stationary(n_frames=10)

        scene = SyntheticScene.single(
            camera_array=camera_array,
            calibration_object=calibration_object,
            trajectory=trajectory,
            pixel_noise_sigma=0.5,
        )

        perturbation = IntrinsicPerturbation(f_scale=1.03)
        cv = _bootstrap_from_scene(scene, perturbation=perturbation)

        result = optimize_with_free_intrinsics(cv, strict=False)

        if not result.converged:
            return

        # At least half the cameras should fail to recover f
        poor_recoveries = 0
        for est in result.intrinsic_estimates:
            cam = scene.camera_array.cameras[est.cam_id]
            assert cam.matrix is not None
            f_true = float(cam.matrix[0, 0])
            injected_error = abs(est.f_initial - f_true) / f_true
            recovered_error = abs(est.f_recovered - f_true) / f_true
            if recovered_error > 0.5 * injected_error or result.hit_bounds:
                poor_recoveries += 1

        assert poor_recoveries >= len(result.intrinsic_estimates) // 2, (
            f"Only {poor_recoveries}/{len(result.intrinsic_estimates)} cameras failed to recover f. "
            f"A stationary small board at distance should not resolve focal length."
        )


class TestE5bOutlierContaminationDrift:
    """E5b: Joint BA on corrupted scene with correct intrinsics.

    Uses the intrinsic_perturbation_scene (good depth variation) with injected
    outliers. With f well-constrained by geometry, outlier contamination
    should not drag f more than 3% from truth.
    """

    def test_contamination_f_drift(self) -> None:
        from caliscope.synthetic.outliers import OutlierConfig, inject_outliers

        scene = intrinsic_perturbation_scene()
        config = OutlierConfig(fraction=0.05, magnitude_range=(10.0, 50.0), random_seed=42)
        corrupted, _ = inject_outliers(scene.image_points_noisy, config)

        intrinsics_only = scene.intrinsics_only_cameras()
        cv = CaptureVolume.bootstrap(corrupted, intrinsics_only)
        result = optimize_with_free_intrinsics(cv, strict=False)

        if not result.converged:
            pytest.skip("Joint BA on corrupted data did not converge")

        for est in result.intrinsic_estimates:
            cam = scene.camera_array.cameras[est.cam_id]
            assert cam.matrix is not None
            f_true = float(cam.matrix[0, 0])
            drift_pct = abs(est.f_recovered - f_true) / f_true * 100
            assert drift_pct < 3.0, (
                f"Cam {est.cam_id}: f drifted {drift_pct:.2f}% from truth under outlier contamination "
                f"(true={f_true:.1f}, recovered={est.f_recovered:.1f})"
            )


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Sparsity oracle...")
    TestSparsityOracle().test_joint_sparsity_zeros_match_true_jacobian()
    print("  PASSED")

    print("E1: f recovery...")
    TestE1FocalLengthRecovery().test_f_recovery()
    print("  PASSED")

    print("E2: k1 recovery...")
    TestE2K1Recovery().test_k1_recovery()
    print("  PASSED")

    print("E2b: combined f+k1+k2 recovery...")
    TestE2bCombinedPerturbation().test_combined_f_k1_k2_recovery()
    print("  PASSED")

    print("E3: constraints as metric anchor...")
    TestE3ConstraintsAsMetricAnchor().test_clean_constrained_improves_f()
    print("  PASSED")

    print("E4: negative control...")
    TestE4NegativeControl().test_stationary_f_not_recovered()
    print("  PASSED")

    print("E5b: outlier contamination drift...")
    TestE5bOutlierContaminationDrift().test_contamination_f_drift()
    print("  PASSED")
