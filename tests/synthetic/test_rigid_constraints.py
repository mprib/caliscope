"""End-to-end tests for rigid constraints using multi-object synthetic scenes.

Exercises the D2 (static markers), D3 (rigid constraint rows in bundle
adjustment), and D5 (origin setting via align_to_object) acceptance criteria
together, through the synthetic testing framework.
"""

from __future__ import annotations

import cv2
import numpy as np

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import STATIC_SYNC_INDEX
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.scene_factories import aruco_scene
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.se3_pose import SE3Pose
from caliscope.synthetic.trajectory import Trajectory


def _make_camera_array() -> CameraArray:
    """4-camera ring, radius 2m, height 0.5m, facing the origin.

    Markers orbit at height 0 (their local Z-normal points along world Z).
    A camera height offset avoids a degenerate edge-on view of that plane.
    """
    return CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()


def _make_aruco_scene() -> tuple[SyntheticScene, ConstraintSet, CameraArray]:
    """Scene with one mobile (orbiting) marker and one static marker."""
    camera_array = _make_camera_array()
    markers = {
        0: ArucoMarker(0, 0.1),  # mobile, 100mm
        1: ArucoMarker(1, 0.1, static=True),  # static, 100mm
    }
    marker_set = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)

    mobile_traj = Trajectory.orbital(n_frames=20, radius=0.5)
    static_pose = SE3Pose.from_axis_angle(
        axis=np.array([0.0, 0.0, 1.0]),
        angle_rad=0.0,
        translation=np.array([0.3, -0.2, 0.0]),
    )
    static_traj = Trajectory.stationary(n_frames=20, pose=static_pose)

    trajectories = {0: mobile_traj, 1: static_traj}
    scene, constraints = aruco_scene(
        marker_set=marker_set,
        trajectories=trajectories,
        camera_array=camera_array,
        pixel_noise_sigma=0.5,
    )
    return scene, constraints, camera_array


class TestStaticMarkerTriangulation:
    """D2: static markers collapse to a single set of world points."""

    def test_static_marker_produces_single_world_point_set(self) -> None:
        scene, constraints, camera_array = _make_aruco_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        capture_volume = CaptureVolume.bootstrap(
            scene.image_points_noisy,
            intrinsics_only,
            constraints=constraints,
        )

        world_df = capture_volume.world_points.df

        static_rows = world_df[(world_df["object_id"] == 1) & (world_df["sync_index"] == STATIC_SYNC_INDEX)]
        assert len(static_rows) == 4, f"Expected 4 static world rows for marker 1, got {len(static_rows)}"

        # Sentinel sync_index is excluded from min/max
        assert capture_volume.world_points.min_index >= 0
        assert capture_volume.world_points.max_index >= capture_volume.world_points.min_index

        mobile_rows = world_df[world_df["object_id"] == 0]
        assert not mobile_rows.empty, "Expected per-frame world rows for the mobile marker"
        assert set(mobile_rows["sync_index"]) != {STATIC_SYNC_INDEX}


class TestConstrainedVsUnconstrainedOptimization:
    """D3: rigid constraints reduce deformation relative to unconstrained BA."""

    def test_constrained_optimization_has_lower_rigidity_error(self) -> None:
        scene, constraints, camera_array = _make_aruco_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        bootstrapped = CaptureVolume.bootstrap(
            scene.image_points_noisy,
            intrinsics_only,
            constraints=constraints,
        )

        constrained = bootstrapped.optimize(use_constraints=True)
        unconstrained = bootstrapped.optimize(use_constraints=False)

        constrained_rmse = constrained.rigidity_report().rmse_mm
        unconstrained_rmse = unconstrained.rigidity_report().rmse_mm

        assert constrained_rmse < unconstrained_rmse, (
            f"Expected constrained RMSE ({constrained_rmse:.3f}mm) < unconstrained RMSE ({unconstrained_rmse:.3f}mm)"
        )


class TestAlignToStaticObject:
    """D5: align_to_object works with a static marker's collapsed world points."""

    def test_align_to_static_marker_succeeds(self) -> None:
        scene, constraints, camera_array = _make_aruco_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        bootstrapped = CaptureVolume.bootstrap(
            scene.image_points_noisy,
            intrinsics_only,
            constraints=constraints,
        )
        optimized = bootstrapped.optimize(use_constraints=True)

        aligned = optimized.align_to_object(sync_index=None, object_id=1)

        assert aligned is not None
        assert not aligned.world_points.df.empty


class TestSparsityOracle:
    """D3 acceptance: sparsity pattern matches the true Jacobian structure.

    Wrong sparsity silently corrupts finite-difference Jacobians. This test
    computes the full dense Jacobian and verifies every zero in the sparsity
    pattern corresponds to a true zero partial derivative.
    """

    def test_sparsity_zeros_match_true_jacobian(self) -> None:
        from caliscope.core.reprojection import bundle_residuals

        scene, constraints, camera_array = _make_aruco_scene()
        intrinsics_only = scene.intrinsics_only_cameras()

        cv = CaptureVolume.bootstrap(
            scene.image_points_noisy,
            intrinsics_only,
            constraints=constraints,
        )

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

        arrays = cv._build_constraint_arrays()
        assert arrays is not None
        c_pairs, c_dists, c_sigmas = arrays

        focal_lengths = [cam.matrix[0, 0] for cam in cv.camera_array.posed_cameras.values() if cam.matrix is not None]
        f_median = float(np.median(focal_lengths))
        c_weights = (1.0 / f_median) / c_sigmas

        x0 = cv._get_vectorized_params()
        sparsity = cv._get_sparsity_pattern(camera_indices, obj_indices, c_pairs)

        def residual_fn(x: np.ndarray) -> np.ndarray:
            return bundle_residuals(
                x,
                cv.camera_array,
                camera_indices,
                image_coords,
                obj_indices,
                True,
                c_pairs,
                c_dists,
                c_weights,
            )

        # Compute dense Jacobian via forward differences
        f0 = residual_fn(x0)
        eps = 1e-7
        n_params = len(x0)
        J = np.zeros((len(f0), n_params))
        for j in range(n_params):
            x_plus = x0.copy()
            x_plus[j] += eps
            J[:, j] = (residual_fn(x_plus) - f0) / eps

        # Every zero in the sparsity pattern should be near-zero in the true Jacobian
        zero_mask = sparsity.toarray() == 0
        max_in_zeros = float(np.max(np.abs(J[zero_mask]))) if zero_mask.any() else 0.0
        assert max_in_zeros < 1e-4, f"Sparsity pattern has false zeros: max |J| in zero entries = {max_in_zeros:.2e}"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Testing static marker triangulation...")
    TestStaticMarkerTriangulation().test_static_marker_produces_single_world_point_set()
    print("  PASSED")

    print("\nTesting constrained vs unconstrained optimization...")
    TestConstrainedVsUnconstrainedOptimization().test_constrained_optimization_has_lower_rigidity_error()
    print("  PASSED")

    print("\nTesting align_to_object with static marker...")
    TestAlignToStaticObject().test_align_to_static_marker_succeeds()
    print("  PASSED")
