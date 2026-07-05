import numpy as np
import pandas as pd
import pytest
import cv2

from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet, DistanceLink
from caliscope.core.constraints import (
    CentroidDistanceConstraint,
    ConstraintSet,
    ConstraintViolation,
    DistanceConstraint,
    RigidityReport,
)
from caliscope.core.point_data import STATIC_SYNC_INDEX, WorldPoints


def test_constraint_set_from_single_marker():
    markers = {0: ArucoMarker(0, 1.0)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs = ConstraintSet.from_marker_set(ms)
    assert len(cs.distances) == 6  # C(4,2) = 6
    assert len(cs.static_object_ids) == 0


def test_constraint_set_from_8_markers_with_corner_link():
    """A corner DistanceLink passes through as exactly ONE DistanceConstraint —
    not one per corner. This is the new explicit-measurement model: the user's
    single measured distance is the constraint, no derived pairs.
    """
    markers = {i: ArucoMarker(i, 1.0, static=(i >= 4)) for i in range(8)}
    link = DistanceLink(marker_a=0, corner_a=1, marker_b=1, corner_b=0, distance_m=0.004)
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers, links=(link,))
    cs = ConstraintSet.from_marker_set(ms)
    assert len(cs.distances) == 8 * 6 + 1  # 48 intra + 1 corner link = 49
    assert len(cs.centroid_distances) == 0
    assert cs.static_object_ids == frozenset({4, 5, 6, 7})

    link_constraint = next(d for d in cs.distances if d.object_id_a == 0 and d.object_id_b == 1)
    assert link_constraint == DistanceConstraint(
        object_id_a=0,
        keypoint_id_a=1,
        object_id_b=1,
        keypoint_id_b=0,
        distance=0.004,
        sigma=0.002,  # default corner sigma
    )


def test_constraint_set_from_marker_set_with_center_link():
    """A center DistanceLink compiles to exactly one CentroidDistanceConstraint,
    leaving the DistanceConstraint list to only the intra-marker geometry.
    """
    markers = {0: ArucoMarker(0, 1.0), 1: ArucoMarker(1, 1.0)}
    link = DistanceLink(marker_a=0, marker_b=1, distance_m=0.512)
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers, links=(link,))
    cs = ConstraintSet.from_marker_set(ms)

    assert len(cs.distances) == 2 * 6  # intra-marker only, no derived corner pairs
    assert cs.centroid_distances == (
        CentroidDistanceConstraint(object_id_a=0, object_id_b=1, distance=0.512, sigma=0.005),
    )


def test_constraint_set_sigma_defaulting():
    """Explicit link sigma_m wins; otherwise corner links default to sigma_m
    and center links default to center_sigma_m (from_marker_set parameters).
    """
    markers = {0: ArucoMarker(0, 1.0), 1: ArucoMarker(1, 1.0)}
    corner_link_default = DistanceLink(marker_a=0, corner_a=0, marker_b=1, corner_b=0, distance_m=0.3)
    corner_link_explicit = DistanceLink(marker_a=0, corner_a=1, marker_b=1, corner_b=1, distance_m=0.3, sigma_m=0.0005)
    ms = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_50,
        markers=markers,
        links=(corner_link_default, corner_link_explicit),
    )
    cs = ConstraintSet.from_marker_set(ms, sigma_m=0.001, center_sigma_m=0.01)
    default_constraint = next(d for d in cs.distances if d.keypoint_id_a == 0 and d.keypoint_id_b == 0)
    explicit_constraint = next(d for d in cs.distances if d.keypoint_id_a == 1 and d.keypoint_id_b == 1)
    assert default_constraint.sigma == pytest.approx(0.001)
    assert explicit_constraint.sigma == pytest.approx(0.0005)

    # Center links: a marker pair can carry only one link (duplicate-endpoint
    # rejection), so default and explicit sigma are exercised on separate sets.
    markers_center = {2: ArucoMarker(2, 1.0), 3: ArucoMarker(3, 1.0)}
    center_link_default = DistanceLink(marker_a=2, marker_b=3, distance_m=0.6)
    ms_center_default = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_50, markers=markers_center, links=(center_link_default,)
    )
    cs_center_default = ConstraintSet.from_marker_set(ms_center_default, sigma_m=0.001, center_sigma_m=0.01)
    assert cs_center_default.centroid_distances[0].sigma == pytest.approx(0.01)

    center_link_explicit = DistanceLink(marker_a=2, marker_b=3, distance_m=0.6, sigma_m=0.02)
    ms_center_explicit = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_50, markers=markers_center, links=(center_link_explicit,)
    )
    cs_center_explicit = ConstraintSet.from_marker_set(ms_center_explicit, sigma_m=0.001, center_sigma_m=0.01)
    assert cs_center_explicit.centroid_distances[0].sigma == pytest.approx(0.02)


def test_constraint_set_distances_correct():
    """Edge distances = size_m, diagonal = size_m * sqrt(2)."""
    markers = {0: ArucoMarker(0, 1.0)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs = ConstraintSet.from_marker_set(ms)
    dists = sorted(d.distance for d in cs.distances)
    # 4 edges of length 1.0, 2 diagonals of length sqrt(2)
    expected = sorted([1.0] * 4 + [np.sqrt(2)] * 2)
    np.testing.assert_allclose(dists, expected, atol=1e-10)


def test_constraint_set_toml_round_trip(tmp_path):
    markers = {
        0: ArucoMarker(0, 1.0),
        1: ArucoMarker(1, 1.0),
        4: ArucoMarker(4, 1.0, static=True),
    }
    center_link = DistanceLink(marker_a=0, marker_b=1, distance_m=0.512, sigma_m=0.005)
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers, links=(center_link,))
    original = ConstraintSet.from_marker_set(ms)
    assert len(original.centroid_distances) == 1  # sanity: exercise the centroid path too

    path = tmp_path / "constraints.toml"
    original.to_toml(path)
    loaded = ConstraintSet.from_toml(path)
    assert len(loaded.distances) == len(original.distances)
    assert len(loaded.centroid_distances) == len(original.centroid_distances)
    assert loaded.static_object_ids == original.static_object_ids
    for orig, load in zip(original.distances, loaded.distances):
        assert load.distance == pytest.approx(orig.distance)
        assert load.sigma == pytest.approx(orig.sigma)
    for orig, load in zip(original.centroid_distances, loaded.centroid_distances):
        assert load.object_id_a == orig.object_id_a
        assert load.object_id_b == orig.object_id_b
        assert load.distance == pytest.approx(orig.distance)
        assert load.sigma == pytest.approx(orig.sigma)


def test_constraint_set_from_toml_missing_centroid_key(tmp_path):
    """A compiled TOML predating centroid_distances (no such key) loads with an
    empty tuple rather than raising — older artifacts stay readable.
    """
    import rtoml

    path = tmp_path / "constraints.toml"
    rtoml.dump(
        {
            "static_object_ids": [],
            "distances": [
                {
                    "object_id_a": 0,
                    "keypoint_id_a": 0,
                    "object_id_b": 0,
                    "keypoint_id_b": 1,
                    "distance": 1.0,
                    "sigma": 0.002,
                }
            ],
        },
        path,
    )
    loaded = ConstraintSet.from_toml(path)
    assert loaded.centroid_distances == ()
    assert len(loaded.distances) == 1


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


def test_joint_residuals_with_constraints():
    """joint_residuals appends constraint rows after reprojection rows."""
    from caliscope.cameras.camera_array import CameraArray, CameraData
    from caliscope.core.bundle_parameterization import BundleParameterization
    from caliscope.core.reprojection import joint_residuals

    cam = CameraData(
        cam_id=0,
        size=(400, 400),
        matrix=np.array([[200, 0, 200], [0, 200, 200], [0, 0, 1]], dtype=np.float64),
        distortions=np.zeros(5),
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 5.0]),
    )
    ca = CameraArray(cameras={0: cam})

    parameterization = BundleParameterization.from_camera_array(ca, n_points=2, refine_intrinsics=False)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    params = parameterization.pack(ca, points)

    camera_indices = np.array([0, 0], dtype=np.int16)
    image_coords = np.array([[200.0, 200.0], [240.0, 200.0]])
    obj_indices = np.array([0, 1], dtype=np.int32)

    # Without constraints
    res_no = joint_residuals(params, parameterization, camera_indices, image_coords, obj_indices)
    assert len(res_no) == 4  # 2 obs × 2

    # With constraints: one distance constraint between points 0 and 1
    c_pairs = np.array([[0, 1]], dtype=np.int32)
    c_dists = np.array([1.0])
    c_weights = np.array([0.5])

    res_yes = joint_residuals(
        params,
        parameterization,
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


def test_filter_preserves_static_world_points():
    """Static world points must survive reprojection filtering even though
    their image observations carry real sync_indices.
    """
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

    # World points:
    # - Static object 10: (0,0,0) and (1,0,0) at STATIC_SYNC_INDEX
    # - Mobile object 0: (0,1,0) and (1,1,0) at sync_index 0 and 1
    world_df = pd.DataFrame(
        {
            "sync_index": [STATIC_SYNC_INDEX, STATIC_SYNC_INDEX, 0, 0, 1, 1],
            "object_id": [10, 10, 0, 0, 0, 0],
            "keypoint_id": [0, 1, 0, 1, 0, 1],
            "x_coord": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y_coord": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "z_coord": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "frame_time": [np.nan, np.nan, 0.0, 0.0, 0.1, 0.1],
        }
    )
    wp = WorldPoints(world_df)

    # Image observations:
    # At sync_index=0: all exact projections → ~0 error
    # At sync_index=1: static obj=10, kp=0 has big offset → ~141px error
    img_df = pd.DataFrame(
        {
            "sync_index": [0, 0, 0, 0, 1, 1, 1, 1],
            "cam_id": [0, 0, 0, 0, 0, 0, 0, 0],
            "object_id": [10, 10, 0, 0, 10, 10, 0, 0],
            "keypoint_id": [0, 1, 0, 1, 0, 1, 0, 1],
            "img_loc_x": [200.0, 240.0, 200.0, 240.0, 300.0, 240.0, 200.0, 240.0],
            "img_loc_y": [200.0, 200.0, 240.0, 240.0, 300.0, 200.0, 240.0, 240.0],
        }
    )
    ip = ImagePoints(img_df)

    # Constraints that declare object 10 as static
    markers = {10: ArucoMarker(10, 1.0, static=True)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
    cs = ConstraintSet.from_marker_set(ms)

    cv = CaptureVolume(camera_array=ca, image_points=ip, world_points=wp, constraints=cs)

    # Pre-filter: 8 observations, 6 world points (2 static + 4 mobile)
    assert len(cv.image_points.df) == 8
    assert len(cv.world_points.df) == 6

    # Filter with 10px threshold drops the bad static observation
    filtered = cv.filter_by_absolute_error(max_pixels=10.0, min_per_camera=1)

    # 7 observations remain, but both static world points survive
    assert len(filtered.image_points.df) == 7
    static_world = filtered.world_points.df[filtered.world_points.df["sync_index"] == STATIC_SYNC_INDEX]
    assert len(static_world) == 2
    assert set(static_world["keypoint_id"]) == {0, 1}

    # All mobile world points also survive
    mobile_world = filtered.world_points.df[filtered.world_points.df["sync_index"] != STATIC_SYNC_INDEX]
    assert len(mobile_world) == 4


def test_rigidity_report_no_longer_has_per_frame_relative_rmse_pct():
    """per_frame_relative_rmse_pct was removed; its only consumer (rigidity sparkline) is gone.

    Other RigidityReport properties must remain intact.
    """
    violations = (
        ConstraintViolation(
            object_id_a=0,
            keypoint_id_a=0,
            object_id_b=0,
            keypoint_id_b=1,
            sync_index=0,
            expected=1.0,
            actual=1.01,
        ),
        ConstraintViolation(
            object_id_a=0,
            keypoint_id_a=0,
            object_id_b=1,
            keypoint_id_b=1,
            sync_index=1,
            expected=2.0,
            actual=1.98,
        ),
    )
    report = RigidityReport(violations=violations)

    assert not hasattr(report, "per_frame_relative_rmse_pct")
    assert report.rmse_mm > 0.0
    assert report.relative_rmse_pct > 0.0
    assert report.max_violation_mm > 0.0
    assert set(report.per_object_rmse_mm.keys()) == {0, 1}
    assert set(report.per_object_relative_rmse_pct.keys()) == {0, 1}
