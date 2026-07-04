from copy import deepcopy

import numpy as np
import pytest

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.bundle_parameterization import BundleParameterization
from caliscope.exceptions import CalibrationError


def _make_camera(cam_id: int, fx: float = 800.0, fy: float = 750.0, fisheye: bool = False) -> CameraData:
    cam = CameraData(cam_id=cam_id, size=(640, 480), fisheye=fisheye)
    cam.matrix = np.array([[fx, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]])
    if fisheye:
        cam.distortions = np.array([0.1, -0.05, 0.01, 0.002])
    else:
        cam.distortions = np.array([-0.2, 0.1, 0.001, -0.001, 0.05])
    cam.rotation = np.eye(3)
    cam.translation = np.array([float(cam_id), 0.0, 0.0])
    return cam


def _make_3cam_array() -> CameraArray:
    cameras = {i: _make_camera(i, fx=800.0 + i * 50, fy=750.0 + i * 50) for i in range(3)}
    return CameraArray(cameras)


class TestPackUnpackRoundTrip:
    def test_identity_round_trip(self):
        array = _make_3cam_array()
        original = deepcopy(array)
        points = np.random.default_rng(42).uniform(-1, 1, (20, 3))

        param = BundleParameterization.from_camera_array(array, n_points=20, refine_intrinsics=True)
        x = param.pack(array, points)
        copy_array = deepcopy(array)
        recovered_points = param.unpack_into(copy_array, x)

        assert np.allclose(recovered_points, points)
        for cam_id in original.cameras:
            orig = original.cameras[cam_id]
            rec = copy_array.cameras[cam_id]
            assert np.allclose(orig.rotation, rec.rotation)
            assert np.allclose(orig.translation, rec.translation)
            assert np.allclose(orig.matrix, rec.matrix)
            assert np.allclose(orig.distortions, rec.distortions)


class TestVariableWidth:
    def test_mixed_free_locked(self):
        cam0 = _make_camera(0)
        cam1 = _make_camera(1)
        cam2 = _make_camera(2, fisheye=True)
        array = CameraArray({0: cam0, 1: cam1, 2: cam2})

        param = BundleParameterization.from_camera_array(array, n_points=10, refine_intrinsics=True)

        assert param.blocks[0].n_params == 9
        assert param.blocks[1].n_params == 9
        assert param.blocks[2].n_params == 6  # fisheye always locked
        assert param.n_camera_params == 9 + 9 + 6
        assert param.camera_param_offsets == (0, 9, 18)


class TestPerturbedUnpack:
    def test_focal_scale_and_k1(self):
        array = _make_3cam_array()
        points = np.random.default_rng(42).uniform(-1, 1, (10, 3))
        param = BundleParameterization.from_camera_array(array, n_points=10, refine_intrinsics=True)
        x = param.pack(array, points)

        # Perturb camera 0: s=1.1, k1=0.05
        off = param.camera_param_offsets[0] + 6
        x[off] = 1.1  # s
        x[off + 1] = 0.05  # k1

        copy_array = deepcopy(array)
        param.unpack_into(copy_array, x)

        block = param.blocks[0]
        cam = copy_array.cameras[0]
        assert np.isclose(cam.matrix[0, 0], 1.1 * block.fx_initial)
        assert np.isclose(cam.matrix[1, 1], 1.1 * block.fy_initial)
        # fx != fy initially, ratio preserved
        assert not np.isclose(block.fx_initial, block.fy_initial)
        assert np.isclose(cam.matrix[0, 0] / cam.matrix[1, 1], block.fx_initial / block.fy_initial)
        assert np.isclose(cam.distortions[0], 0.05)
        # p1, p2, k3 unchanged
        assert np.allclose(cam.distortions[2:5], list(block.dist_fixed))


class TestBounds:
    def test_shapes_and_values(self):
        array = _make_3cam_array()
        param = BundleParameterization.from_camera_array(array, n_points=15, refine_intrinsics=True)
        lower, upper = param.bounds()

        n_total = param.n_camera_params + 3 * 15
        assert lower.shape == (n_total,)
        assert upper.shape == (n_total,)

        # Extrinsic params unbounded
        assert lower[0] == -np.inf
        assert upper[0] == np.inf

        # Free intrinsic bounds for camera 0
        off = param.camera_param_offsets[0] + 6
        assert lower[off] == 0.5
        assert upper[off] == 2.0
        assert lower[off + 1] == -1.0
        assert upper[off + 1] == 1.0
        assert lower[off + 2] == -2.0
        assert upper[off + 2] == 2.0


class TestBoundWarnings:
    def test_fires_at_boundary(self):
        array = _make_3cam_array()
        points = np.zeros((5, 3))
        param = BundleParameterization.from_camera_array(array, n_points=5, refine_intrinsics=True)
        x = param.pack(array, points)

        # Put camera 0 s at lower bound
        off0 = param.camera_param_offsets[0] + 6
        x[off0] = 0.5

        # Put camera 1 k1 at upper bound
        off1 = param.camera_param_offsets[1] + 6
        x[off1 + 1] = 1.0

        warnings = param.bound_warnings(x)
        cam_ids = [w.cam_id for w in warnings]
        params = [w.parameter for w in warnings]
        assert 0 in cam_ids
        assert 1 in cam_ids
        assert "f" in params
        assert "k1" in params

    def test_no_warning_at_interior(self):
        array = _make_3cam_array()
        points = np.zeros((5, 3))
        param = BundleParameterization.from_camera_array(array, n_points=5, refine_intrinsics=True)
        x = param.pack(array, points)
        # s=1.0, k1/k2 at initial values — all interior
        warnings = param.bound_warnings(x)
        assert len(warnings) == 0


class TestFisheye:
    def test_invalid_distortion_count_raises(self):
        cam = CameraData(cam_id=0, size=(640, 480), fisheye=True)
        cam.matrix = np.eye(3) * 500
        cam.matrix[2, 2] = 1.0
        cam.distortions = np.array([0.1, -0.05, 0.01, 0.002, 0.0])  # 5 coeffs
        cam.rotation = np.eye(3)
        cam.translation = np.zeros(3)
        array = CameraArray({0: cam})

        with pytest.raises(CalibrationError, match="exactly 4 distortion coefficients"):
            BundleParameterization.from_camera_array(array, n_points=1, refine_intrinsics=True)

    def test_valid_fisheye_gives_locked_block(self):
        cam = _make_camera(0, fisheye=True)
        array = CameraArray({0: cam})
        param = BundleParameterization.from_camera_array(array, n_points=1, refine_intrinsics=True)
        assert param.blocks[0].free_intrinsics is False
        assert param.blocks[0].n_params == 6


class TestMissingIntrinsics:
    def test_raises_calibration_error(self):
        cam = CameraData(cam_id=0, size=(640, 480))
        cam.rotation = np.eye(3)
        cam.translation = np.zeros(3)
        array = CameraArray({0: cam})

        with pytest.raises(CalibrationError, match="has no intrinsics"):
            BundleParameterization.from_camera_array(array, n_points=1, refine_intrinsics=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
