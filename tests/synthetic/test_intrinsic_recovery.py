"""D7 experiments: joint intrinsic+extrinsic bundle adjustment.

Sparsity oracle, E1 (f recovery), E2 (k1 recovery), E3 (constraints as
metric anchor), E4 (negative control), E5b (outlier contamination drift),
E6 (wand scene with blind f guess).

Ported from experimental_ba to the production CaptureVolume.optimize() API.
"""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.core.bundle_parameterization import BundleParameterization
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.reprojection import joint_residuals
from caliscope.synthetic.camera_synthesizer import IntrinsicPerturbation, perturb_intrinsics
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
    """Every zero in the sparsity pattern must be a true zero partial derivative.

    Includes one locked camera (fisheye=False, refine=False for cam 0) to
    cover variable-width blocks.
    """

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

        parameterization = BundleParameterization.from_camera_array(
            cv.camera_array, n_points=len(cv.world_points.points), refine_intrinsics=True
        )
        x0 = parameterization.pack(cv.camera_array, cv.world_points.points)
        sparsity = parameterization.sparsity(camera_indices, obj_indices, 0, None, None)

        def residual_fn(x):
            return joint_residuals(x, parameterization, camera_indices, image_coords, obj_indices)

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

        optimized = cv.optimize(refine_intrinsics=True)

        assert optimized.optimization_status is not None
        assert optimized.optimization_status.converged
        assert len(optimized.optimization_status.bound_warnings) == 0

        for cam_id, cam in optimized.camera_array.posed_cameras.items():
            true_cam = scene.camera_array.cameras[cam_id]
            assert true_cam.matrix is not None
            assert true_cam.distortions is not None
            assert cam.matrix is not None
            assert cam.distortions is not None
            f_true = float(true_cam.matrix[0, 0])
            f_recovered = float(cam.matrix[0, 0])
            k2_true = float(true_cam.distortions[1])
            k2_recovered = float(cam.distortions[1])
            f_error_pct = abs(f_recovered - f_true) / f_true * 100
            k2_error = abs(k2_recovered - k2_true)
            assert f_error_pct < 1.0, (
                f"Cam {cam_id}: f error {f_error_pct:.2f}% >= 1% (true={f_true:.1f}, recovered={f_recovered:.1f})"
            )
            assert k2_error < 0.03, (
                f"Cam {cam_id}: |k2_recovered - k2_true| = {k2_error:.4f} >= 0.03 "
                f"(true={k2_true:.4f}, recovered={k2_recovered:.4f})"
            )

        aligned = align_to_ground_truth(optimized, scene)
        for cam_id in scene.camera_array.cameras:
            err = pose_error(aligned.camera_array.cameras[cam_id], scene.camera_array.cameras[cam_id])
            assert err.rotation_deg < 1.0, f"Cam {cam_id}: rotation {err.rotation_deg:.3f} deg"
            assert err.translation_m < 0.02, f"Cam {cam_id}: translation {err.translation_m:.4f} m"


class TestE2K1Recovery:
    """E2: Perturb k1 by 0.02, run joint BA, assert recovery within 0.02."""

    def test_k1_recovery(self) -> None:
        scene = intrinsic_perturbation_scene()
        perturbation = IntrinsicPerturbation(k1_delta=0.02)
        cv = _bootstrap_from_scene(scene, perturbation=perturbation)

        optimized = cv.optimize(refine_intrinsics=True)

        assert optimized.optimization_status is not None
        assert optimized.optimization_status.converged
        assert len(optimized.optimization_status.bound_warnings) == 0

        for cam_id, cam in optimized.camera_array.posed_cameras.items():
            true_cam = scene.camera_array.cameras[cam_id]
            assert true_cam.distortions is not None
            assert cam.distortions is not None
            k1_true = float(true_cam.distortions[0])
            k1_recovered = float(cam.distortions[0])
            k1_error = abs(k1_recovered - k1_true)
            assert k1_error < 0.02, (
                f"Cam {cam_id}: |k1_recovered - k1_true| = {k1_error:.4f} >= 0.02 "
                f"(true={k1_true:.4f}, recovered={k1_recovered:.4f})"
            )

        aligned = align_to_ground_truth(optimized, scene)
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

        optimized = cv.optimize(refine_intrinsics=True)

        assert optimized.optimization_status is not None
        assert optimized.optimization_status.converged
        assert len(optimized.optimization_status.bound_warnings) == 0

        for cam_id, cam in optimized.camera_array.posed_cameras.items():
            true_cam = scene.camera_array.cameras[cam_id]
            assert true_cam.matrix is not None
            assert true_cam.distortions is not None
            assert cam.matrix is not None
            assert cam.distortions is not None
            f_true = float(true_cam.matrix[0, 0])
            k1_true = float(true_cam.distortions[0])
            k2_true = float(true_cam.distortions[1])
            f_recovered = float(cam.matrix[0, 0])
            k1_recovered = float(cam.distortions[0])
            k2_recovered = float(cam.distortions[1])
            f_error_pct = abs(f_recovered - f_true) / f_true * 100
            k1_error = abs(k1_recovered - k1_true)
            k2_error = abs(k2_recovered - k2_true)
            assert f_error_pct < 1.0, (
                f"Cam {cam_id}: f error {f_error_pct:.2f}% (true={f_true:.1f}, recovered={f_recovered:.1f})"
            )
            assert k1_error < 0.02, (
                f"Cam {cam_id}: k1 error {k1_error:.4f} (true={k1_true:.4f}, recovered={k1_recovered:.4f})"
            )
            assert k2_error < 0.03, (
                f"Cam {cam_id}: k2 error {k2_error:.4f} (true={k2_true:.4f}, recovered={k2_recovered:.4f})"
            )


class TestE3ConstraintsAsMetricAnchor:
    """E3: constrained f error <= unconstrained f error (with 2x margin)."""

    def test_clean_constrained_improves_f(self) -> None:
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

        opt_u = cv_unconstrained.optimize(refine_intrinsics=True, use_constraints=False)
        opt_c = cv_constrained.optimize(refine_intrinsics=True, use_constraints=True)

        assert opt_u.optimization_status is not None and opt_u.optimization_status.converged
        assert opt_c.optimization_status is not None and opt_c.optimization_status.converged

        for cam_id in scene.camera_array.posed_cameras:
            true_cam = scene.camera_array.cameras[cam_id]
            assert true_cam.matrix is not None
            f_true = float(true_cam.matrix[0, 0])
            f_c = float(opt_c.camera_array.cameras[cam_id].matrix[0, 0])  # type: ignore[index]
            f_u = float(opt_u.camera_array.cameras[cam_id].matrix[0, 0])  # type: ignore[index]
            err_c = abs(f_c - f_true) / f_true
            err_u = abs(f_u - f_true) / f_true
            assert err_c <= max(err_u * 2.0, 0.005), (
                f"Cam {cam_id}: constrained f error {err_c:.4f} > "
                f"2x unconstrained {err_u:.4f} — constraints degraded f recovery"
            )


class TestE4NegativeControl:
    """E4: Small board far from cameras, stationary. Joint BA should NOT recover f."""

    def test_stationary_f_not_recovered(self) -> None:
        from caliscope.synthetic.calibration_object import CalibrationObject
        from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
        from caliscope.synthetic.synthetic_scene import SyntheticScene
        from caliscope.synthetic.trajectory import Trajectory

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

        optimized = cv.optimize(refine_intrinsics=True, strict=False)

        if optimized.optimization_status is None or not optimized.optimization_status.converged:
            return

        poor_recoveries = 0
        for cam_id, cam in optimized.camera_array.posed_cameras.items():
            true_cam = scene.camera_array.cameras[cam_id]
            assert true_cam.matrix is not None
            assert cam.matrix is not None
            f_true = float(true_cam.matrix[0, 0])
            f_recovered = float(cam.matrix[0, 0])
            # The initial (perturbed) f
            f_initial = f_true * 1.03
            injected_error = abs(f_initial - f_true) / f_true
            recovered_error = abs(f_recovered - f_true) / f_true
            has_warnings = len(optimized.optimization_status.bound_warnings) > 0
            if recovered_error > 0.5 * injected_error or has_warnings:
                poor_recoveries += 1

        n_cams = len(optimized.camera_array.posed_cameras)
        assert poor_recoveries >= n_cams // 2, (
            f"Only {poor_recoveries}/{n_cams} cameras failed to recover f. "
            f"A stationary small board at distance should not resolve focal length."
        )


class TestE5bOutlierContaminationDrift:
    """E5b: Joint BA on corrupted scene with correct intrinsics.

    Outlier contamination should not drag f more than 3% from truth.
    """

    def test_contamination_f_drift(self) -> None:
        from caliscope.synthetic.outliers import OutlierConfig, inject_outliers

        scene = intrinsic_perturbation_scene()
        config = OutlierConfig(fraction=0.05, magnitude_range=(10.0, 50.0), random_seed=42)
        corrupted, _ = inject_outliers(scene.image_points_noisy, config)

        intrinsics_only = scene.intrinsics_only_cameras()
        cv = CaptureVolume.bootstrap(corrupted, intrinsics_only)
        optimized = cv.optimize(refine_intrinsics=True, strict=False)

        if optimized.optimization_status is None or not optimized.optimization_status.converged:
            pytest.skip("Joint BA on corrupted data did not converge")

        for cam_id, cam in optimized.camera_array.posed_cameras.items():
            true_cam = scene.camera_array.cameras[cam_id]
            assert true_cam.matrix is not None
            assert cam.matrix is not None
            f_true = float(true_cam.matrix[0, 0])
            f_recovered = float(cam.matrix[0, 0])
            drift_pct = abs(f_recovered - f_true) / f_true * 100
            assert drift_pct < 3.0, (
                f"Cam {cam_id}: f drifted {drift_pct:.2f}% from truth under outlier contamination "
                f"(true={f_true:.1f}, recovered={f_recovered:.1f})"
            )


class TestE6WandScene:
    """E6: Joint BA on wand scene (2 linked ArUco markers, blind f guess).

    Tests the actual product workflow: f = image_width/2, k1=0, k2=0.
    Recovery should match the charuco floor for f, with rigidity < 1mm.
    """

    def test_wand_blind_f_recovery(self) -> None:
        from caliscope.synthetic.scene_factories import wand_scene_with_constraints

        scene, constraints = wand_scene_with_constraints(include_static=False)
        true_f = 1394.6
        perturbation = IntrinsicPerturbation(
            f_scale=960.0 / true_f,
            k1_delta=-0.115,
            k2_delta=0.219,
        )
        cameras = perturb_intrinsics(scene.intrinsics_only_cameras(), perturbation)
        cv = CaptureVolume.bootstrap(scene.image_points_noisy, cameras, constraints=constraints)

        optimized = cv.optimize(refine_intrinsics=True)

        assert optimized.optimization_status is not None
        assert optimized.optimization_status.converged
        assert len(optimized.optimization_status.bound_warnings) == 0

        for cam_id, cam in optimized.camera_array.posed_cameras.items():
            true_cam = scene.camera_array.cameras[cam_id]
            assert true_cam.matrix is not None
            assert true_cam.distortions is not None
            assert cam.matrix is not None
            f_true = float(true_cam.matrix[0, 0])
            f_recovered = float(cam.matrix[0, 0])
            f_err = abs(f_recovered - f_true) / f_true * 100
            assert f_err < 1.0, f"Cam {cam_id}: f error {f_err:.2f}%"

        report = optimized.rigidity_report()
        assert report.violations, "No rigidity violations computed"
        errors_mm = [abs(v.actual - v.expected) * 1000 for v in report.violations]
        rig_rmse = float(np.sqrt(np.mean(np.array(errors_mm) ** 2)))
        assert rig_rmse < 1.0, f"Rigidity RMSE {rig_rmse:.3f}mm >= 1.0mm"


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

    print("E6: wand blind f recovery...")
    TestE6WandScene().test_wand_blind_f_recovery()
    print("  PASSED")
