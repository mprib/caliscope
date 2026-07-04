"""Tests for calibration target factories."""

import cv2
import numpy as np
import pytest

from caliscope.core.aruco_marker import ArucoMarker
from caliscope.synthetic.target_factories import (
    aruco_marker,
    charuco_board,
    double_sided_charuco_board,
)
from caliscope.synthetic.scene_factories import (
    aruco_scene,
)
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.trajectory import Trajectory


class TestCharucoBoard:
    def test_point_count(self) -> None:
        obj = charuco_board(5, 7, 0.03)
        assert obj.n_points == 24  # (5-1)*(7-1)

    def test_corner_positions(self) -> None:
        obj = charuco_board(5, 7, 0.03)
        np.testing.assert_allclose(obj.points[0], [0.03, 0.03, 0.0])
        np.testing.assert_allclose(obj.points[23], [0.18, 0.12, 0.0])

    def test_matches_opencv_charuco_board(self) -> None:
        rows, cols, sq = 5, 7, 0.03
        obj = charuco_board(rows, cols, sq)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        cv_board = cv2.aruco.CharucoBoard((cols, rows), sq, sq * 0.5, aruco_dict)
        cv_corners = cv_board.getChessboardCorners()

        # OpenCV may return (N,3) or (N,1,3)
        if cv_corners.ndim == 3:
            cv_corners = cv_corners.reshape(-1, 3)

        np.testing.assert_allclose(obj.points, cv_corners, atol=1e-10)

    def test_single_sided_has_face_normal(self) -> None:
        obj = charuco_board(5, 7, 0.03, single_sided=True)
        assert obj.face_normal is not None
        np.testing.assert_allclose(obj.face_normal, [0.0, 0.0, 1.0])

    def test_double_sided_no_face_normal(self) -> None:
        obj = charuco_board(5, 7, 0.03, single_sided=False)
        assert obj.face_normal is None

    def test_various_board_sizes(self) -> None:
        for r, c in [(3, 4), (6, 9), (8, 11)]:
            obj = charuco_board(r, c, 0.05)
            assert obj.n_points == (r - 1) * (c - 1)


class TestDoubleSidedCharucoBoard:
    def test_matches_charuco_board_single_sided_false(self) -> None:
        a = double_sided_charuco_board(5, 7, 0.03)
        b = charuco_board(5, 7, 0.03, single_sided=False)
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.keypoint_ids, b.keypoint_ids)
        assert a.face_normal is None


class TestArucoMarker:
    def test_matches_production_corners(self) -> None:
        obj = aruco_marker(0.1)
        expected = ArucoMarker(marker_id=0, size_m=0.1).corners
        np.testing.assert_allclose(obj.points, expected)

    def test_single_sided_has_face_normal(self) -> None:
        obj = aruco_marker(0.1, single_sided=True)
        assert obj.face_normal is not None

    def test_double_sided_no_face_normal(self) -> None:
        obj = aruco_marker(0.1, single_sided=False)
        assert obj.face_normal is None

    def test_four_points(self) -> None:
        obj = aruco_marker(0.2)
        assert obj.n_points == 4


class TestVisibilityCulling:
    def test_double_sided_charuco_visible_from_both_sides(self) -> None:
        obj = double_sided_charuco_board(5, 7, 0.05)
        camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
        trajectory = Trajectory.orbital(n_frames=5, radius=0.2)

        scene = SyntheticScene.single(
            camera_array=camera_array,
            calibration_object=obj,
            trajectory=trajectory,
            pixel_noise_sigma=0.0,
        )
        ip = scene.image_points_perfect
        cam_ids_with_obs = ip.df["cam_id"].unique()
        assert len(cam_ids_with_obs) == 4

    def test_single_sided_charuco_invisible_from_back(self) -> None:
        from caliscope.synthetic.scene_factories import _look_at_camera

        obj = charuco_board(3, 4, 0.05, single_sided=True)

        cam_front = _look_at_camera(np.array([0.1, 0.1, 1.5]), np.array([0.1, 0.1, 0.0]), cam_id=0)
        cam_back = _look_at_camera(np.array([0.1, 0.1, -1.5]), np.array([0.1, 0.1, 0.0]), cam_id=1)

        from caliscope.cameras.camera_array import CameraArray

        cameras = CameraArray(cameras={0: cam_front, 1: cam_back})
        scene = SyntheticScene.single(
            camera_array=cameras,
            calibration_object=obj,
            trajectory=Trajectory.stationary(n_frames=1),
            pixel_noise_sigma=0.0,
        )
        ip = scene.image_points_perfect
        cam_ids = ip.df["cam_id"].unique()
        assert 0 in cam_ids
        assert 1 not in cam_ids


class TestArucoSceneSingleSided:
    def test_default_not_single_sided(self) -> None:
        import cv2 as cv2_mod
        from caliscope.core.aruco_marker import ArucoMarkerSet

        markers = {0: ArucoMarker(0, 0.1)}
        marker_set = ArucoMarkerSet(dictionary=cv2_mod.aruco.DICT_4X4_50, markers=markers)
        trajectories = {0: Trajectory.orbital(n_frames=5, radius=0.2)}
        camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()

        scene, _ = aruco_scene(
            marker_set=marker_set,
            trajectories=trajectories,
            camera_array=camera_array,
            single_sided=False,
        )
        cal_obj = scene.objects[0].calibration_object
        assert cal_obj.face_normal is None

    def test_single_sided_has_face_normal(self) -> None:
        import cv2 as cv2_mod
        from caliscope.core.aruco_marker import ArucoMarkerSet

        markers = {0: ArucoMarker(0, 0.1)}
        marker_set = ArucoMarkerSet(dictionary=cv2_mod.aruco.DICT_4X4_50, markers=markers)
        trajectories = {0: Trajectory.orbital(n_frames=5, radius=0.2)}
        camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()

        scene, _ = aruco_scene(
            marker_set=marker_set,
            trajectories=trajectories,
            camera_array=camera_array,
            single_sided=True,
        )
        cal_obj = scene.objects[0].calibration_object
        assert cal_obj.face_normal is not None


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    obj = charuco_board(5, 7, 0.03)
    print(f"Charuco 5x7: {obj.n_points} corners")
    print(f"  Corner 0: {obj.points[0]}")
    print(f"  Corner 23: {obj.points[23]}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    cv_board = cv2.aruco.CharucoBoard((7, 5), 0.03, 0.015, aruco_dict)
    cv_corners = cv_board.getChessboardCorners()
    if cv_corners.ndim == 3:
        cv_corners = cv_corners.reshape(-1, 3)
    print(f"  OpenCV match: {np.allclose(obj.points, cv_corners)}")

    pytest.main([__file__, "-v"])
