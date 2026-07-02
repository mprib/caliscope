import numpy as np
import pytest
import cv2

from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet, MarkerLink
from caliscope.core.constraints import ConstraintSet


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
