"""Correctness gate for the analytic bundle-adjustment Jacobian.

joint_jacobian must agree with central finite differences of joint_residuals
on every parameterization path: pinhole-locked, pinhole-refine, fisheye
(mixed with pinhole to cover heterogeneous block widths), and constraint rows
(corner and centroid endpoint groups). A sign or scale error in any block
shows up here as an O(1) relative disagreement.
"""

from __future__ import annotations

import cv2
import numpy as np

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.bundle_parameterization import BundleParameterization
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.reprojection import joint_jacobian, joint_residuals
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.trajectory import Trajectory


def _finite_difference_jacobian(fun, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    f0 = fun(x0)
    J = np.zeros((len(f0), len(x0)))
    for j in range(len(x0)):
        x_plus = x0.copy()
        x_plus[j] += eps
        x_minus = x0.copy()
        x_minus[j] -= eps
        J[:, j] = (fun(x_plus) - fun(x_minus)) / (2 * eps)
    return J


def _assert_jacobians_match(analytic: np.ndarray, fd: np.ndarray) -> None:
    """Per-column comparison scaled by the column's magnitude.

    Central FD is accurate to ~1e-10 here, so 1e-6 relative catches any real
    derivative error (a sign flip is 2.0, a missing 1/fx scale is ~1000).
    """
    diff = np.abs(analytic - fd)
    col_scale = np.maximum(np.abs(fd).max(axis=0), 1e-3)
    per_column = diff.max(axis=0) / col_scale
    worst_col = int(np.argmax(per_column))
    assert per_column[worst_col] < 1e-6, (
        f"Analytic Jacobian disagrees with finite differences: column {worst_col} "
        f"relative error {per_column[worst_col]:.2e} (abs {diff[:, worst_col].max():.2e})"
    )


def _small_pinhole_scene() -> SyntheticScene:
    camera_array = CameraSynthesizer().add_ring(n=3, radius=2.0, height=0.3).build()
    calibration_object = CalibrationObject.planar_grid(rows=3, cols=4, spacing=0.05)
    trajectory = Trajectory.linear(
        n_frames=8,
        start=np.array([0.3, -0.3, -0.2]),
        end=np.array([-0.3, 0.3, 0.5]),
        tumble_rate=2.0,
        origin_frame=0,
    )
    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=0.5,
    )


def _ba_arrays(cv: CaptureVolume) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (camera_indices, image_coords, obj_indices) the way optimize() does."""
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
    return camera_indices, image_coords, obj_indices


def _scene_jacobian_pair(refine_intrinsics: bool) -> tuple[np.ndarray, np.ndarray]:
    scene = _small_pinhole_scene()
    cv = CaptureVolume.bootstrap(scene.image_points_noisy, scene.intrinsics_only_cameras())
    camera_indices, image_coords, obj_indices = _ba_arrays(cv)

    parameterization = BundleParameterization.from_camera_array(
        cv.camera_array, n_points=len(cv.world_points.points), refine_intrinsics=refine_intrinsics
    )
    x0 = parameterization.pack(cv.camera_array, cv.world_points.points)

    def fun(x):
        return joint_residuals(x, parameterization, camera_indices, image_coords, obj_indices)

    analytic = joint_jacobian(x0, parameterization, camera_indices, image_coords, obj_indices).toarray()
    fd = _finite_difference_jacobian(fun, x0)
    return analytic, fd


class TestReprojectionJacobian:
    def test_pinhole_locked_matches_fd(self) -> None:
        analytic, fd = _scene_jacobian_pair(refine_intrinsics=False)
        _assert_jacobians_match(analytic, fd)

    def test_pinhole_refine_matches_fd(self) -> None:
        analytic, fd = _scene_jacobian_pair(refine_intrinsics=True)
        _assert_jacobians_match(analytic, fd)

    def test_mixed_fisheye_pinhole_matches_fd(self) -> None:
        """Fisheye block (6 params, locked) alongside a free pinhole block (9 params).

        Covers the fisheye Jacobian column slicing and heterogeneous block offsets.
        """
        rvec0, tvec0 = np.array([0.1, -0.05, 0.02]), np.array([0.0, 0.1, 3.0])
        K0 = np.array([[600.0, 0, 320], [0, 590.0, 240], [0, 0, 1]])
        dist0 = np.array([0.1, -0.05, 0.01, 0.002])

        rvec1, tvec1 = np.array([-0.08, 0.12, -0.04]), np.array([0.5, -0.1, 3.2])
        K1 = np.array([[610.0, 0, 315], [0, 605.0, 245], [0, 0, 1]])
        dist1 = np.array([0.08, -0.03, 0.001, -0.002, 0.005])

        camera_array = CameraArray(
            {
                0: CameraData(
                    cam_id=0,
                    size=(640, 480),
                    fisheye=True,
                    matrix=K0,
                    distortions=dist0,
                    rotation=cv2.Rodrigues(rvec0)[0],
                    translation=tvec0,
                ),
                1: CameraData(
                    cam_id=1,
                    size=(640, 480),
                    matrix=K1,
                    distortions=dist1,
                    rotation=cv2.Rodrigues(rvec1)[0],
                    translation=tvec1,
                ),
            }
        )

        rng = np.random.default_rng(42)
        points = rng.uniform(-0.6, 0.6, (25, 3))
        pts = points.reshape(-1, 1, 3)

        proj0, _ = cv2.fisheye.projectPoints(pts, rvec0.reshape(3, 1), tvec0.reshape(3, 1), K0, dist0.reshape(4, 1))
        proj1, _ = cv2.projectPoints(pts, rvec1, tvec1, K1, dist1)
        exact = np.vstack([proj0.reshape(-1, 2), proj1.reshape(-1, 2)])

        image_coords = exact + rng.normal(0, 0.5, exact.shape)
        camera_indices = np.repeat(np.array([0, 1], dtype=np.int16), len(points))
        obj_indices = np.tile(np.arange(len(points), dtype=np.int32), 2)

        parameterization = BundleParameterization.from_camera_array(
            camera_array, n_points=len(points), refine_intrinsics=True
        )
        assert parameterization.blocks[0].n_params == 6
        assert parameterization.blocks[1].n_params == 9
        x0 = parameterization.pack(camera_array, points)

        def fun(x):
            return joint_residuals(x, parameterization, camera_indices, image_coords, obj_indices)

        analytic = joint_jacobian(x0, parameterization, camera_indices, image_coords, obj_indices).toarray()
        fd = _finite_difference_jacobian(fun, x0)
        _assert_jacobians_match(analytic, fd)


class TestConstraintRowJacobian:
    def test_constraint_rows_match_fd(self) -> None:
        """Corner endpoints (one row repeated 4x) and centroid endpoints (4 distinct rows)."""
        scene = _small_pinhole_scene()
        cv = CaptureVolume.bootstrap(scene.image_points_noisy, scene.intrinsics_only_cameras())
        camera_indices, image_coords, obj_indices = _ba_arrays(cv)

        parameterization = BundleParameterization.from_camera_array(
            cv.camera_array, n_points=len(cv.world_points.points), refine_intrinsics=False
        )
        x0 = parameterization.pack(cv.camera_array, cv.world_points.points)

        groups_a = np.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=np.int32)
        groups_b = np.array([[5, 5, 5, 5], [8, 9, 10, 11]], dtype=np.int32)
        distances = np.array([0.11, 0.07])
        weights = np.array([2.0, 3.5])

        def fun(x):
            return joint_residuals(
                x,
                parameterization,
                camera_indices,
                image_coords,
                obj_indices,
                groups_a,
                groups_b,
                distances,
                weights,
            )

        analytic = joint_jacobian(
            x0,
            parameterization,
            camera_indices,
            image_coords,
            obj_indices,
            groups_a,
            groups_b,
            distances,
            weights,
        ).toarray()
        fd = _finite_difference_jacobian(fun, x0)

        n_reproj_rows = 2 * len(camera_indices)
        assert analytic.shape[0] == n_reproj_rows + 2
        _assert_jacobians_match(analytic, fd)
        # The constraint rows specifically must carry signal, not just pass inside
        # reprojection-dominated column scales.
        constraint_diff = np.abs(analytic[n_reproj_rows:] - fd[n_reproj_rows:])
        assert constraint_diff.max() < 1e-8, f"Constraint row max abs error {constraint_diff.max():.2e}"
        assert np.abs(analytic[n_reproj_rows:]).max() > 0.1, "Constraint rows are unexpectedly all near zero"


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Pinhole locked...")
    TestReprojectionJacobian().test_pinhole_locked_matches_fd()
    print("  PASSED")

    print("Pinhole refine...")
    TestReprojectionJacobian().test_pinhole_refine_matches_fd()
    print("  PASSED")

    print("Mixed fisheye + pinhole...")
    TestReprojectionJacobian().test_mixed_fisheye_pinhole_matches_fd()
    print("  PASSED")

    print("Constraint rows...")
    TestConstraintRowJacobian().test_constraint_rows_match_fd()
    print("  PASSED")
