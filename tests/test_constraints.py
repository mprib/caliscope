import numpy as np
import pandas as pd
import pytest
import cv2

from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet, DistanceLink
from caliscope.core.charuco import Charuco
from caliscope.core.constraints import (
    CentroidDistanceConstraint,
    ConstraintSet,
    ConstraintViolation,
    DistanceConstraint,
    RigidityReport,
)
from caliscope.core.point_data import STATIC_SYNC_INDEX, WorldPoints


def _identity_cam_array():
    """A single camera at z=+5 looking down -Z, no distortion. Enough to satisfy
    CaptureVolume geometry checks in constraint-focused tests.
    """
    from caliscope.cameras.camera_array import CameraArray, CameraData

    cam = CameraData(
        cam_id=0,
        size=(400, 400),
        matrix=np.array([[200, 0, 200], [0, 200, 200], [0, 0, 1]], dtype=np.float64),
        distortions=np.zeros(5),
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 5.0]),
    )
    return CameraArray(cameras={0: cam})


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


def test_constraint_set_toml_round_trip_no_centroids(tmp_path):
    """A charuco-shaped ConstraintSet — non-empty distances and
    static_object_ids, but NO centroid_distances — must round-trip. rtoml
    raises TomlSerializationError if an empty list key is written after a
    non-empty array-of-tables key ("values must be emitted before tables"),
    so centroid_distances must be omitted entirely when empty.
    """
    distances = (
        DistanceConstraint(object_id_a=0, keypoint_id_a=0, object_id_b=0, keypoint_id_b=1, distance=1.0, sigma=0.002),
        DistanceConstraint(object_id_a=0, keypoint_id_a=1, object_id_b=0, keypoint_id_b=2, distance=2.0, sigma=0.002),
    )
    original = ConstraintSet(
        distances=distances,
        static_object_ids=frozenset({4, 7}),
        centroid_distances=(),
    )

    path = tmp_path / "constraints.toml"
    original.to_toml(path)

    assert "centroid_distances" not in path.read_text()

    loaded = ConstraintSet.from_toml(path)
    assert loaded.distances == original.distances
    assert loaded.static_object_ids == original.static_object_ids
    assert loaded.centroid_distances == ()


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
    groups_a, groups_b, dists, sigmas = result
    # 6 constraints × 3 frames = 18 instances
    assert groups_a.shape == (18, 4)
    assert groups_b.shape == (18, 4)
    assert len(dists) == 18
    assert len(sigmas) == 18
    # Corner endpoints are one row index repeated 4x (mean recovers the point exactly)
    assert np.all(groups_a[:, 0:1] == groups_a)
    assert np.all(groups_b[:, 0:1] == groups_b)


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

    # With constraints: one corner distance constraint between points 0 and 1.
    # Corner endpoints are width-4 groups of one repeated row index.
    c_groups_a = np.array([[0, 0, 0, 0]], dtype=np.int32)
    c_groups_b = np.array([[1, 1, 1, 1]], dtype=np.int32)
    c_dists = np.array([1.0])
    c_weights = np.array([0.5])

    res_yes = joint_residuals(
        params,
        parameterization,
        camera_indices,
        image_coords,
        obj_indices,
        constraint_groups_a=c_groups_a,
        constraint_groups_b=c_groups_b,
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


# -- Endpoint groups: corner exact-equality + centroid path --


def _build_parameterization(ca, n_points):
    from caliscope.core.bundle_parameterization import BundleParameterization

    return BundleParameterization.from_camera_array(ca, n_points=n_points, refine_intrinsics=False)


def test_joint_residuals_corner_group_exact_equality():
    """A corner endpoint is one row index repeated 4x. The mean of four identical
    rows is bitwise-identical to the row (verified in review), so the grouped
    residual EXACTLY equals the pre-groups single-index residual — assert ==.
    """
    from caliscope.core.reprojection import joint_residuals

    ca = _identity_cam_array()
    parameterization = _build_parameterization(ca, n_points=2)
    points = np.array([[0.1, 0.2, 0.3], [1.4, -0.5, 0.7]])
    params = parameterization.pack(ca, points)

    camera_indices = np.array([0, 0], dtype=np.int16)
    image_coords = np.array([[200.0, 200.0], [240.0, 200.0]])
    obj_indices = np.array([0, 1], dtype=np.int32)

    c_dists = np.array([1.0])
    c_weights = np.array([0.5])
    groups_a = np.array([[0, 0, 0, 0]], dtype=np.int32)
    groups_b = np.array([[1, 1, 1, 1]], dtype=np.int32)

    res = joint_residuals(
        params,
        parameterization,
        camera_indices,
        image_coords,
        obj_indices,
        constraint_groups_a=groups_a,
        constraint_groups_b=groups_b,
        constraint_distances=c_dists,
        constraint_weights=c_weights,
    )

    # Pre-groups reference: single-index difference, no averaging.
    expected = (np.linalg.norm(points[0] - points[1]) - c_dists[0]) * c_weights[0]
    assert res[-1] == expected  # exact, not approx


def test_joint_residuals_centroid_zero_at_exact_geometry():
    """A centroid endpoint averages a marker's four corner rows. When the two
    centroids sit exactly the declared distance apart, the residual is zero.
    """
    from caliscope.core.reprojection import joint_residuals

    ca = _identity_cam_array()
    # Marker 0 corners → centroid (0,0,0); marker 1 corners → centroid (2,0,0).
    pts = np.array(
        [
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, -0.5, 0.0],
            [1.5, 0.5, 0.0],
            [2.5, 0.5, 0.0],
            [2.5, -0.5, 0.0],
            [1.5, -0.5, 0.0],
        ]
    )
    parameterization = _build_parameterization(ca, n_points=8)
    params = parameterization.pack(ca, pts)

    camera_indices = np.zeros(8, dtype=np.int16)
    image_coords = np.tile(np.array([200.0, 200.0]), (8, 1))
    obj_indices = np.arange(8, dtype=np.int32)

    groups_a = np.array([[0, 1, 2, 3]], dtype=np.int32)
    groups_b = np.array([[4, 5, 6, 7]], dtype=np.int32)
    c_dists = np.array([2.0])
    c_weights = np.array([1.0])

    res = joint_residuals(
        params,
        parameterization,
        camera_indices,
        image_coords,
        obj_indices,
        constraint_groups_a=groups_a,
        constraint_groups_b=groups_b,
        constraint_distances=c_dists,
        constraint_weights=c_weights,
    )
    assert res[-1] == pytest.approx(0.0, abs=1e-12)


def _two_marker_world_and_image(centroid_b_x: float, sync_indices=(0,)):
    """Two 4-corner markers per sync_index. Marker 0 centroid at origin, marker 1
    centroid at (centroid_b_x, 0, 0). Returns (world_df, img_df).
    """
    corner_offsets = [(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    world_rows = []
    img_rows = []
    for si in sync_indices:
        for obj_id, cx in ((0, 0.0), (1, centroid_b_x)):
            for kid, (dx, dy) in enumerate(corner_offsets):
                world_rows.append(
                    {
                        "sync_index": si,
                        "object_id": obj_id,
                        "keypoint_id": kid,
                        "x_coord": cx + dx,
                        "y_coord": dy,
                        "z_coord": 0.0,
                        "frame_time": float(si) * 0.1,
                    }
                )
                img_rows.append(
                    {
                        "sync_index": si,
                        "cam_id": 0,
                        "object_id": obj_id,
                        "keypoint_id": kid,
                        "img_loc_x": 200.0 + cx * 10 + dx,
                        "img_loc_y": 200.0 + dy,
                    }
                )
    return pd.DataFrame(world_rows), pd.DataFrame(img_rows)


def test_rigidity_report_centroid_violation_hand_computed():
    """A centroid constraint whose actual centroid separation differs from the
    declared distance produces one centroid-kind violation with the hand-computed
    magnitude and keypoint ids of -1 (no single corner to name).
    """
    from caliscope.core.point_data import ImagePoints
    from caliscope.core.capture_volume import CaptureVolume

    # Marker 1 centroid actually at x=2.0; declare the distance as 1.5.
    world_df, img_df = _two_marker_world_and_image(centroid_b_x=2.0)
    cs = ConstraintSet(
        distances=(),
        static_object_ids=frozenset(),
        centroid_distances=(CentroidDistanceConstraint(object_id_a=0, object_id_b=1, distance=1.5, sigma=0.005),),
    )
    cv = CaptureVolume(
        camera_array=_identity_cam_array(),
        image_points=ImagePoints(img_df),
        world_points=WorldPoints(world_df),
        constraints=cs,
    )

    report = cv.rigidity_report()
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.kind == "centroid"
    assert v.keypoint_id_a == -1
    assert v.keypoint_id_b == -1
    assert v.object_id_a == 0
    assert v.object_id_b == 1
    assert v.expected == pytest.approx(1.5)
    assert v.actual == pytest.approx(2.0)


def test_build_constraint_arrays_centroid_fires_only_with_all_8_corners():
    """A centroid constraint fires only in sync_indices where all eight corner
    rows (four per marker) are present.
    """
    from caliscope.core.point_data import ImagePoints
    from caliscope.core.capture_volume import CaptureVolume

    # Two frames' worth of geometry; then remove one corner of marker 1 in sync 1.
    world_df, img_df = _two_marker_world_and_image(centroid_b_x=2.0, sync_indices=(0, 1))
    drop = (world_df["sync_index"] == 1) & (world_df["object_id"] == 1) & (world_df["keypoint_id"] == 3)
    world_df = world_df[~drop].reset_index(drop=True)
    img_drop = (img_df["sync_index"] == 1) & (img_df["object_id"] == 1) & (img_df["keypoint_id"] == 3)
    img_df = img_df[~img_drop].reset_index(drop=True)

    cs = ConstraintSet(
        distances=(),
        static_object_ids=frozenset(),
        centroid_distances=(CentroidDistanceConstraint(object_id_a=0, object_id_b=1, distance=2.0, sigma=0.005),),
    )
    cv = CaptureVolume(
        camera_array=_identity_cam_array(),
        image_points=ImagePoints(img_df),
        world_points=WorldPoints(world_df),
        constraints=cs,
    )

    result = cv._build_constraint_arrays()
    assert result is not None
    groups_a, groups_b, dists, sigmas = result
    # Only sync 0 has all 8 corners → exactly one centroid instance.
    assert groups_a.shape == (1, 4)
    assert groups_b.shape == (1, 4)
    assert len(dists) == 1


def test_build_constraint_arrays_static_static_centroid_fires_once():
    """A centroid link between two static markers fires exactly once, at
    STATIC_SYNC_INDEX, when all eight static corner rows exist.
    """
    from caliscope.core.point_data import ImagePoints
    from caliscope.core.capture_volume import CaptureVolume

    corner_offsets = [(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    world_rows = []
    for obj_id, cx in ((0, 0.0), (1, 2.0)):
        for kid, (dx, dy) in enumerate(corner_offsets):
            world_rows.append(
                {
                    "sync_index": STATIC_SYNC_INDEX,
                    "object_id": obj_id,
                    "keypoint_id": kid,
                    "x_coord": cx + dx,
                    "y_coord": dy,
                    "z_coord": 0.0,
                    "frame_time": np.nan,
                }
            )
    # Image observations carry real sync_indices even for static markers.
    img_rows = []
    for si in (0, 1):
        for obj_id, cx in ((0, 0.0), (1, 2.0)):
            for kid, (dx, dy) in enumerate(corner_offsets):
                img_rows.append(
                    {
                        "sync_index": si,
                        "cam_id": 0,
                        "object_id": obj_id,
                        "keypoint_id": kid,
                        "img_loc_x": 200.0 + cx * 10 + dx,
                        "img_loc_y": 200.0 + dy,
                    }
                )

    cs = ConstraintSet(
        distances=(),
        static_object_ids=frozenset({0, 1}),
        centroid_distances=(CentroidDistanceConstraint(object_id_a=0, object_id_b=1, distance=2.0, sigma=0.005),),
    )
    cv = CaptureVolume(
        camera_array=_identity_cam_array(),
        image_points=ImagePoints(pd.DataFrame(img_rows)),
        world_points=WorldPoints(pd.DataFrame(world_rows)),
        constraints=cs,
    )

    result = cv._build_constraint_arrays()
    assert result is not None
    groups_a, groups_b, dists, sigmas = result
    assert groups_a.shape == (1, 4)
    assert len(dists) == 1


def test_sparsity_marks_at_most_24_columns_per_constraint_row():
    """Each constraint row marks 3 coordinate columns per distinct endpoint row:
    a corner constraint touches 2 distinct rows (6 columns); a centroid constraint
    touches 8 distinct rows (24 columns, the maximum).
    """
    ca = _identity_cam_array()
    parameterization = _build_parameterization(ca, n_points=8)

    camera_indices = np.zeros(2, dtype=np.int16)
    obj_indices = np.array([0, 1], dtype=np.int32)

    # Row 0: corner constraint (repeated indices). Row 1: centroid (8 distinct rows).
    groups_a = np.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=np.int32)
    groups_b = np.array([[1, 1, 1, 1], [4, 5, 6, 7]], dtype=np.int32)

    sp = parameterization.sparsity(camera_indices, obj_indices, 2, groups_a, groups_b)

    n_obs = len(camera_indices)
    corner_row = sp.getrow(n_obs * 2 + 0).toarray().sum()
    centroid_row = sp.getrow(n_obs * 2 + 1).toarray().sum()
    assert corner_row == 6
    assert centroid_row == 24
    assert corner_row <= 24 and centroid_row <= 24


def test_static_marker_guard_gates_on_intra_only_rmse():
    """The static-marker guard must gate on per-object RMSE computed from
    intra-marker violations only. A static-static center-link disagreement
    (a cross-object violation) alone must NOT trip the guard: with perfect
    intra geometry, the intra-only RMSE stays zero even though the general
    per_object_rmse_mm (which attributes cross-object violations to both
    endpoints) is large.
    """
    from caliscope.core.point_data import ImagePoints
    from caliscope.core.capture_volume import CaptureVolume

    # Two static markers, perfect intra geometry (unit squares), but their
    # centroids sit 2.0 apart while a static-static center link declares 0.5 —
    # a gross tape-measure disagreement that must not drag a rigid marker down.
    corner_offsets = [(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    world_rows = []
    for obj_id, cx in ((0, 0.0), (1, 2.0)):
        for kid, (dx, dy) in enumerate(corner_offsets):
            world_rows.append(
                {
                    "sync_index": STATIC_SYNC_INDEX,
                    "object_id": obj_id,
                    "keypoint_id": kid,
                    "x_coord": cx + dx,
                    "y_coord": dy,
                    "z_coord": 0.0,
                    "frame_time": np.nan,
                }
            )
    img_rows = []
    for obj_id, cx in ((0, 0.0), (1, 2.0)):
        for kid, (dx, dy) in enumerate(corner_offsets):
            img_rows.append(
                {
                    "sync_index": 0,
                    "cam_id": 0,
                    "object_id": obj_id,
                    "keypoint_id": kid,
                    "img_loc_x": 200.0 + cx * 10 + dx,
                    "img_loc_y": 200.0 + dy,
                }
            )

    # Perfect intra distances for a unit square (edges 1.0, diagonals sqrt(2)).
    intra = []
    for obj_id in (0, 1):
        corners = np.array([(dx, dy, 0.0) for dx, dy in corner_offsets])
        for i in range(4):
            for j in range(i + 1, 4):
                intra.append(
                    DistanceConstraint(
                        object_id_a=obj_id,
                        keypoint_id_a=i,
                        object_id_b=obj_id,
                        keypoint_id_b=j,
                        distance=float(np.linalg.norm(corners[i] - corners[j])),
                        sigma=0.002,
                    )
                )
    cs = ConstraintSet(
        distances=tuple(intra),
        static_object_ids=frozenset({0, 1}),
        centroid_distances=(CentroidDistanceConstraint(object_id_a=0, object_id_b=1, distance=0.5, sigma=0.005),),
    )
    cv = CaptureVolume(
        camera_array=_identity_cam_array(),
        image_points=ImagePoints(pd.DataFrame(img_rows)),
        world_points=WorldPoints(pd.DataFrame(world_rows)),
        constraints=cs,
    )

    report = cv.rigidity_report()
    max_intra_mm = np.sqrt(2) * 1000.0
    threshold = 0.25 * max_intra_mm

    # General per-object RMSE folds in the cross-link violation → trips old guard.
    assert report.per_object_rmse_mm[0] > threshold

    # Intra-only per-object RMSE (the corrected gate) stays at zero → no drop.
    intra_violations = tuple(v for v in report.violations if v.object_id_a == v.object_id_b)
    intra_rmse = RigidityReport(violations=intra_violations).per_object_rmse_mm.get(0, 0.0)
    assert intra_rmse < threshold
    assert intra_rmse == pytest.approx(0.0, abs=1e-6)


# -- ConstraintSet.from_charuco --


def _five_by_seven_charuco() -> Charuco:
    """5x7-square board -> 4x6 corner grid (both dims >= 3), the case the
    count formula requires.
    """
    return Charuco.from_squares(columns=7, rows=5, square_size_cm=5.0)


def test_from_charuco_constraint_count_matches_formula():
    """R(C-1) horizontal + (R-1)C vertical + 2(R-1)(C-1) diagonals + 6 braces,
    for an R x C corner grid with both dims >= 3 (extreme corners then don't
    alias truss edges).
    """
    charuco = _five_by_seven_charuco()
    corners = np.asarray(charuco.board.getChessboardCorners())
    n_cols = len(set(np.round(corners[:, 0] / charuco.board.getSquareLength()).astype(int)))
    n_rows = len(set(np.round(corners[:, 1] / charuco.board.getSquareLength()).astype(int)))
    assert n_rows >= 3 and n_cols >= 3

    cs = ConstraintSet.from_charuco(charuco)

    expected = n_rows * (n_cols - 1) + (n_rows - 1) * n_cols + 2 * (n_rows - 1) * (n_cols - 1) + 6
    assert len(cs.distances) == expected


def test_from_charuco_all_distances_positive():
    charuco = _five_by_seven_charuco()
    cs = ConstraintSet.from_charuco(charuco)
    assert len(cs.distances) > 0
    assert all(d.distance > 0 for d in cs.distances)


def test_from_charuco_neighbor_distances_equal_square_length():
    """Horizontal and vertical adjacent-corner distances equal the board's
    square_length exactly (within a tight float tolerance) — they are the
    literal edges of the printed grid squares.
    """
    charuco = _five_by_seven_charuco()
    square_length = float(charuco.board.getSquareLength())
    cs = ConstraintSet.from_charuco(charuco)

    neighbor_distances = [d.distance for d in cs.distances if d.distance == pytest.approx(square_length, abs=1e-9)]
    # every row contributes (n_cols - 1) horizontal edges, every column (n_rows - 1)
    # vertical edges; all of them measure exactly one square_length.
    assert len(neighbor_distances) > 0
    for dist in neighbor_distances:
        assert abs(dist - square_length) < 1e-9


def test_from_charuco_braces_present():
    """The 4 extreme board corners (min/max x times min/max y) are all
    pairwise braced (6 distances among them), guarding against paper-fold
    degeneracy along interior grid lines.
    """
    charuco = _five_by_seven_charuco()
    corners = np.asarray(charuco.board.getChessboardCorners())
    xs, ys = corners[:, 0], corners[:, 1]

    def _closest(target_x, target_y):
        return int(np.argmin((xs - target_x) ** 2 + (ys - target_y) ** 2))

    extreme_ids = {
        _closest(xs.min(), ys.min()),
        _closest(xs.min(), ys.max()),
        _closest(xs.max(), ys.min()),
        _closest(xs.max(), ys.max()),
    }
    assert len(extreme_ids) == 4

    cs = ConstraintSet.from_charuco(charuco)
    brace_pairs = {
        frozenset((d.keypoint_id_a, d.keypoint_id_b))
        for d in cs.distances
        if d.keypoint_id_a in extreme_ids and d.keypoint_id_b in extreme_ids
    }
    # C(4, 2) = 6 pairwise braces among the 4 extreme corners.
    assert len(brace_pairs) == 6


def test_from_charuco_object_ids_all_zero_and_no_centroids():
    """object_id is 0 for every corner (matching CharucoTracker's identity
    scheme); static_object_ids is empty (the board moves); no centroid
    constraints are ever produced from a charuco board.
    """
    charuco = _five_by_seven_charuco()
    cs = ConstraintSet.from_charuco(charuco)

    assert all(d.object_id_a == 0 and d.object_id_b == 0 for d in cs.distances)
    assert cs.static_object_ids == frozenset()
    assert cs.centroid_distances == ()
