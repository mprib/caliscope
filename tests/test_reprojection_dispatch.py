"""Tests for project_points dispatch and joint_residuals."""

import cv2
import numpy as np
import pytest

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.bundle_parameterization import BundleParameterization
from caliscope.core.reprojection import joint_residuals, project_points


class TestProjectPointsFisheye:
    def test_fisheye_matches_opencv_directly(self):
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        D = np.array([0.1, -0.05, 0.01, 0.0])
        rvec = np.array([0.1, -0.05, 0.02])
        tvec = np.array([0.0, 0.0, 5.0])

        rng = np.random.default_rng(42)
        points = rng.uniform(-1, 1, (20, 3))
        points[:, 2] += 3.0  # ensure in front of camera

        result = project_points(points, rvec, tvec, K, D, fisheye=True)

        expected, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3), rvec.reshape(3, 1), tvec.reshape(3, 1), K, D.reshape(4, 1)
        )
        expected = expected.reshape(-1, 2)

        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_fisheye_differs_from_brown_conrady(self):
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        D = np.array([0.1, -0.05, 0.01, 0.0])
        rvec = np.array([0.1, -0.05, 0.02])
        tvec = np.array([0.0, 0.0, 5.0])

        rng = np.random.default_rng(42)
        points = rng.uniform(-0.5, 0.5, (20, 3))
        points[:, 2] += 3.0

        fisheye_result = project_points(points, rvec, tvec, K, D, fisheye=True)
        # Brown-Conrady needs 5 coeffs
        bc_dist = np.array([0.1, -0.05, 0.01, 0.0, 0.0])
        bc_result = project_points(points, rvec, tvec, K, bc_dist, fisheye=False)

        assert not np.allclose(fisheye_result, bc_result, atol=0.1)

    def test_fisheye_5coef_raises(self):
        K = np.eye(3) * 500
        D = np.zeros(5)
        with pytest.raises(ValueError, match="4 distortion coefficients"):
            project_points(np.zeros((1, 3)), np.zeros(3), np.zeros(3), K, D, fisheye=True)


class TestJointResiduals:
    def test_zero_residual_at_exact_projection(self):
        """When observations are exact projections, residuals are zero."""
        cam0 = CameraData(cam_id=0, size=(640, 480))
        cam0.matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        cam0.distortions = np.zeros(5)
        cam0.rotation = np.eye(3)
        cam0.translation = np.array([0.0, 0.0, 0.0])

        cam1 = CameraData(cam_id=1, size=(640, 480))
        cam1.matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        cam1.distortions = np.zeros(5)
        cam1.rotation = np.eye(3)
        cam1.translation = np.array([1.0, 0.0, 0.0])

        ca = CameraArray({0: cam0, 1: cam1})
        parameterization = BundleParameterization.from_camera_array(ca, n_points=3, refine_intrinsics=False)

        points = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [-1.0, 1.0, 5.0]])

        # Generate exact observations
        image_coords_list = []
        camera_indices_list = []
        obj_indices_list = []
        for cam_idx, cam in enumerate([cam0, cam1]):
            rvec = cv2.Rodrigues(cam.rotation)[0].ravel()
            for pt_idx, pt in enumerate(points):
                proj, _ = cv2.projectPoints(pt.reshape(1, 1, 3), rvec, cam.translation, cam.matrix, cam.distortions)
                image_coords_list.append(proj.reshape(2))
                camera_indices_list.append(cam_idx)
                obj_indices_list.append(pt_idx)

        image_coords = np.array(image_coords_list)
        camera_indices = np.array(camera_indices_list, dtype=np.int16)
        obj_indices = np.array(obj_indices_list, dtype=np.int32)

        x = parameterization.pack(ca, points)
        residuals = joint_residuals(x, parameterization, camera_indices, image_coords, obj_indices)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)

    def test_perturbed_point_only_affects_its_rows(self):
        """Perturbing one world point only changes that point's residual rows."""
        cam0 = CameraData(cam_id=0, size=(640, 480))
        cam0.matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        cam0.distortions = np.zeros(5)
        cam0.rotation = np.eye(3)
        cam0.translation = np.zeros(3)

        ca = CameraArray({0: cam0})
        points = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]])
        parameterization = BundleParameterization.from_camera_array(ca, n_points=2, refine_intrinsics=False)

        # Generate exact observations
        rvec = np.zeros(3)
        obs = []
        for pt in points:
            proj, _ = cv2.projectPoints(pt.reshape(1, 1, 3), rvec, np.zeros(3), cam0.matrix, cam0.distortions)
            obs.append(proj.reshape(2))
        image_coords = np.array(obs)
        camera_indices = np.array([0, 0], dtype=np.int16)
        obj_indices = np.array([0, 1], dtype=np.int32)

        x = parameterization.pack(ca, points)
        r0 = joint_residuals(x, parameterization, camera_indices, image_coords, obj_indices)
        np.testing.assert_allclose(r0, 0.0, atol=1e-10)

        # Perturb point 1
        x_perturbed = x.copy()
        pt_offset = parameterization.n_camera_params + 3  # start of point 1
        x_perturbed[pt_offset] += 0.5

        r1 = joint_residuals(x_perturbed, parameterization, camera_indices, image_coords, obj_indices)
        # Point 0 rows (indices 0,1) should still be zero
        np.testing.assert_allclose(r1[0:2], 0.0, atol=1e-10)
        # Point 1 rows (indices 2,3) should be nonzero
        assert np.any(np.abs(r1[2:4]) > 1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
