import numpy as np
import pandas as pd
import pytest
import cv2

from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet, MarkerLink
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import STATIC_SYNC_INDEX, WorldPoints


def test_constraint_set_from_single_marker():
    markers = {0: ArucoMarker(0, 1.0)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs = ConstraintSet.from_marker_set(ms)
    assert len(cs.distances) == 6  # C(4,2) = 6
    assert len(cs.static_object_ids) == 0


def test_constraint_set_from_8_markers_with_link():
    markers = {i: ArucoMarker(i, 1.0, static=(i >= 4)) for i in range(8)}
    link = MarkerLink(marker_a=0, marker_b=1, corner_map=(1, 0, 3, 2), separation_m=0.004)
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers, links=(link,))
    cs = ConstraintSet.from_marker_set(ms)
    assert len(cs.distances) == 8 * 6 + 4  # 48 intra + 4 link = 52
    assert cs.static_object_ids == frozenset({4, 5, 6, 7})


def test_constraint_set_distances_correct():
    """Edge distances = size_m, diagonal = size_m * sqrt(2)."""
    markers = {0: ArucoMarker(0, 1.0)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs = ConstraintSet.from_marker_set(ms)
    dists = sorted(d.distance for d in cs.distances)
    # 4 edges of length 1.0, 2 diagonals of length sqrt(2)
    expected = sorted([1.0] * 4 + [np.sqrt(2)] * 2)
    np.testing.assert_allclose(dists, expected, atol=1e-10)


def test_constraint_set_unit_scale():
    markers = {0: ArucoMarker(0, 1.0)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs_m = ConstraintSet.from_marker_set(ms, unit_scale=1.0)
    cs_mm = ConstraintSet.from_marker_set(ms, unit_scale=1000.0)
    for d_m, d_mm in zip(cs_m.distances, cs_mm.distances):
        assert d_mm.distance == pytest.approx(d_m.distance * 1000.0)
        assert d_mm.sigma == pytest.approx(d_m.sigma * 1000.0)


def test_constraint_set_toml_round_trip(tmp_path):
    markers = {0: ArucoMarker(0, 1.0), 4: ArucoMarker(4, 1.0, static=True)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    original = ConstraintSet.from_marker_set(ms)
    path = tmp_path / "constraints.toml"
    original.to_toml(path)
    loaded = ConstraintSet.from_toml(path)
    assert len(loaded.distances) == len(original.distances)
    assert loaded.static_object_ids == original.static_object_ids
    for orig, load in zip(original.distances, loaded.distances):
        assert load.distance == pytest.approx(orig.distance)
        assert load.sigma == pytest.approx(orig.sigma)


def test_constraint_set_from_toml_missing(tmp_path):
    from caliscope.persistence import PersistenceError

    with pytest.raises(PersistenceError):
        ConstraintSet.from_toml(tmp_path / "nope.toml")


def test_world_points_static_sync_index_excluded_from_min_max():
    df = pd.DataFrame(
        {
            "sync_index": [STATIC_SYNC_INDEX, STATIC_SYNC_INDEX, 10, 20, 30],
            "object_id": [4, 4, 0, 0, 0],
            "keypoint_id": [0, 1, 0, 0, 0],
            "x_coord": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y_coord": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z_coord": [1.0, 2.0, 3.0, 4.0, 5.0],
            "frame_time": [np.nan, np.nan, 0.1, 0.2, 0.3],
        }
    )
    wp = WorldPoints(df)
    assert wp.min_index == 10
    assert wp.max_index == 30


def test_world_points_all_static():
    df = pd.DataFrame(
        {
            "sync_index": [STATIC_SYNC_INDEX, STATIC_SYNC_INDEX],
            "object_id": [4, 4],
            "keypoint_id": [0, 1],
            "x_coord": [1.0, 2.0],
            "y_coord": [1.0, 2.0],
            "z_coord": [1.0, 2.0],
            "frame_time": [np.nan, np.nan],
        }
    )
    wp = WorldPoints(df)
    assert wp.min_index == 0
    assert wp.max_index == 0


# -- D3: Constraint rows in BA --


def test_rigidity_report_perfect_square():
    """A perfect 1m square has zero violations."""
    markers = {0: ArucoMarker(0, 1.0)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs = ConstraintSet.from_marker_set(ms)

    s = 0.5
    world_df = pd.DataFrame(
        {
            "sync_index": [0, 0, 0, 0],
            "object_id": [0, 0, 0, 0],
            "keypoint_id": [0, 1, 2, 3],
            "x_coord": [-s, s, s, -s],
            "y_coord": [s, s, -s, -s],
            "z_coord": [0.0, 0.0, 0.0, 0.0],
            "frame_time": [0.0, 0.0, 0.0, 0.0],
        }
    )
    img_df = pd.DataFrame(
        {
            "sync_index": [0, 0, 0, 0],
            "cam_id": [0, 0, 0, 0],
            "object_id": [0, 0, 0, 0],
            "keypoint_id": [0, 1, 2, 3],
            "img_loc_x": [100.0, 200.0, 200.0, 100.0],
            "img_loc_y": [100.0, 100.0, 200.0, 200.0],
        }
    )

    from caliscope.core.point_data import ImagePoints

    wp = WorldPoints(world_df)
    ip = ImagePoints(img_df)

    from caliscope.core.capture_volume import CaptureVolume
    from caliscope.cameras.camera_array import CameraArray, CameraData

    cam = CameraData(
        cam_id=0,
        size=(400, 400),
        matrix=np.array([[200, 0, 200], [0, 200, 200], [0, 0, 1]], dtype=np.float64),
        distortions=np.zeros(5),
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 5.0]),
    )
    ca = CameraArray(cameras={0: cam})
    cv = CaptureVolume(camera_array=ca, image_points=ip, world_points=wp, constraints=cs)

    report = cv.rigidity_report()
    assert len(report.violations) == 6
    assert report.rmse_mm == pytest.approx(0.0, abs=1e-6)


def test_rigidity_report_deformed_square():
    """A deformed square produces nonzero violations."""
    markers = {0: ArucoMarker(0, 1.0)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs = ConstraintSet.from_marker_set(ms)

    s = 0.5
    world_df = pd.DataFrame(
        {
            "sync_index": [0, 0, 0, 0],
            "object_id": [0, 0, 0, 0],
            "keypoint_id": [0, 1, 2, 3],
            "x_coord": [-s, s + 0.01, s, -s],  # corner 1 shifted 10mm
            "y_coord": [s, s, -s, -s],
            "z_coord": [0.0, 0.0, 0.0, 0.0],
            "frame_time": [0.0, 0.0, 0.0, 0.0],
        }
    )
    img_df = pd.DataFrame(
        {
            "sync_index": [0, 0, 0, 0],
            "cam_id": [0, 0, 0, 0],
            "object_id": [0, 0, 0, 0],
            "keypoint_id": [0, 1, 2, 3],
            "img_loc_x": [100.0, 200.0, 200.0, 100.0],
            "img_loc_y": [100.0, 100.0, 200.0, 200.0],
        }
    )

    from caliscope.core.point_data import ImagePoints
    from caliscope.core.capture_volume import CaptureVolume
    from caliscope.cameras.camera_array import CameraArray, CameraData

    cam = CameraData(
        cam_id=0,
        size=(400, 400),
        matrix=np.array([[200, 0, 200], [0, 200, 200], [0, 0, 1]], dtype=np.float64),
        distortions=np.zeros(5),
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 5.0]),
    )
    ca = CameraArray(cameras={0: cam})
    wp = WorldPoints(world_df)
    ip = ImagePoints(img_df)
    cv = CaptureVolume(camera_array=ca, image_points=ip, world_points=wp, constraints=cs)

    report = cv.rigidity_report()
    assert report.rmse_mm > 1.0
    assert report.max_violation_mm > 5.0


def test_build_constraint_arrays_mobile_marker():
    """Constraint arrays for a mobile marker produce correct instance count."""
    markers = {0: ArucoMarker(0, 1.0)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs = ConstraintSet.from_marker_set(ms)

    # 3 frames, 4 corners each
    rows = []
    for si in [0, 1, 2]:
        for kid in range(4):
            rows.append(
                {
                    "sync_index": si,
                    "object_id": 0,
                    "keypoint_id": kid,
                    "x_coord": float(kid),
                    "y_coord": 0.0,
                    "z_coord": 0.0,
                    "frame_time": si * 0.1,
                }
            )
    world_df = pd.DataFrame(rows)

    img_rows = []
    for si in [0, 1, 2]:
        for kid in range(4):
            img_rows.append(
                {
                    "sync_index": si,
                    "cam_id": 0,
                    "object_id": 0,
                    "keypoint_id": kid,
                    "img_loc_x": 100.0 + kid * 50,
                    "img_loc_y": 100.0,
                }
            )
    img_df = pd.DataFrame(img_rows)

    from caliscope.core.point_data import ImagePoints
    from caliscope.core.capture_volume import CaptureVolume
    from caliscope.cameras.camera_array import CameraArray, CameraData

    cam = CameraData(
        cam_id=0,
        size=(400, 400),
        matrix=np.array([[200, 0, 200], [0, 200, 200], [0, 0, 1]], dtype=np.float64),
        distortions=np.zeros(5),
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 5.0]),
    )
    ca = CameraArray(cameras={0: cam})
    wp = WorldPoints(world_df)
    ip = ImagePoints(img_df)
    cv = CaptureVolume(camera_array=ca, image_points=ip, world_points=wp, constraints=cs)

    result = cv._build_constraint_arrays()
    assert result is not None
    pairs, dists, sigmas = result
    # 6 constraints × 3 frames = 18 instances
    assert len(pairs) == 18
    assert len(dists) == 18
    assert len(sigmas) == 18


def test_bundle_residuals_with_constraints():
    """bundle_residuals appends constraint rows after reprojection rows."""
    from caliscope.core.reprojection import bundle_residuals
    from caliscope.cameras.camera_array import CameraArray, CameraData

    cam = CameraData(
        cam_id=0,
        size=(400, 400),
        matrix=np.array([[200, 0, 200], [0, 200, 200], [0, 0, 1]], dtype=np.float64),
        distortions=np.zeros(5),
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 5.0]),
    )
    ca = CameraArray(cameras={0: cam})

    # 2 points, 2 observations
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    params = np.concatenate([np.array([0, 0, 0, 0, 0, 5.0]), points.ravel()])

    camera_indices = np.array([0, 0], dtype=np.int16)
    image_coords = np.array([[200.0, 200.0], [240.0, 200.0]])
    obj_indices = np.array([0, 1], dtype=np.int32)

    # Without constraints
    res_no = bundle_residuals(params, ca, camera_indices, image_coords, obj_indices)
    assert len(res_no) == 4  # 2 obs × 2

    # With constraints: one distance constraint between points 0 and 1
    c_pairs = np.array([[0, 1]], dtype=np.int32)
    c_dists = np.array([1.0])
    c_weights = np.array([0.5])

    res_yes = bundle_residuals(
        params,
        ca,
        camera_indices,
        image_coords,
        obj_indices,
        constraint_pairs=c_pairs,
        constraint_distances=c_dists,
        constraint_weights=c_weights,
    )
    assert len(res_yes) == 5  # 4 reproj + 1 constraint
    # Reprojection rows are identical
    np.testing.assert_allclose(res_yes[:4], res_no)
    # Constraint residual: (||[1,0,0]-[0,0,0]|| - 1.0) * 0.5 = 0.0
    assert res_yes[4] == pytest.approx(0.0, abs=1e-10)
