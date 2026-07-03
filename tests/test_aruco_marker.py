import numpy as np
import pytest
import cv2

from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet, MarkerLink


# -- ArucoMarker --


def test_aruco_marker_static_default():
    marker = ArucoMarker(marker_id=0, size_m=0.10)
    assert marker.static is False


def test_aruco_marker_static_true():
    marker = ArucoMarker(marker_id=0, size_m=0.10, static=True)
    assert marker.static is True


def test_aruco_marker_corners_from_size():
    marker = ArucoMarker(marker_id=0, size_m=0.10)
    corners = marker.corners
    assert corners.shape == (4, 3)
    s = 0.05
    expected = np.array([[-s, +s, 0], [+s, +s, 0], [+s, -s, 0], [-s, -s, 0]])
    np.testing.assert_allclose(corners, expected)


def test_aruco_marker_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="positive"):
        ArucoMarker(marker_id=0, size_m=0.0)
    with pytest.raises(ValueError, match="positive"):
        ArucoMarker(marker_id=0, size_m=-0.05)


def test_aruco_marker_frozen():
    marker = ArucoMarker(marker_id=0, size_m=0.05)
    with pytest.raises(AttributeError):
        marker.size_m = 0.1  # type: ignore[misc]


def test_aruco_marker_corners_cached():
    marker = ArucoMarker(marker_id=0, size_m=0.05)
    assert marker.corners is marker.corners


# -- ArucoMarkerSet --


def test_marker_set_construction():
    markers = {0: ArucoMarker(0, 0.165), 3: ArucoMarker(3, 0.10)}
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_100, markers=markers)
    assert len(ms.markers) == 2
    assert ms.markers[0].size_m == 0.165
    assert ms.markers[3].size_m == 0.10


def test_marker_set_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers={})


def test_marker_set_rejects_overcapacity_id():
    with pytest.raises(ValueError, match="capacity"):
        ArucoMarkerSet(
            dictionary=cv2.aruco.DICT_4X4_50,
            markers={99: ArucoMarker(99, 0.05)},
        )


def test_marker_set_toml_round_trip(tmp_path):
    markers = {
        0: ArucoMarker(0, 0.165),
        3: ArucoMarker(3, 0.10),
        7: ArucoMarker(7, 0.165),
    }
    original = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_100, markers=markers)
    path = tmp_path / "aruco_marker_set.toml"
    original.to_toml(path)
    loaded = ArucoMarkerSet.from_toml(path)
    assert loaded.dictionary == original.dictionary
    assert len(loaded.markers) == 3
    for mid in [0, 3, 7]:
        np.testing.assert_allclose(loaded.markers[mid].corners, original.markers[mid].corners)
        assert loaded.markers[mid].size_m == original.markers[mid].size_m


def test_marker_set_from_toml_missing_file(tmp_path):
    from caliscope.persistence import PersistenceError

    with pytest.raises(PersistenceError):
        ArucoMarkerSet.from_toml(tmp_path / "nope.toml")


def test_marker_set_toml_round_trip_with_static(tmp_path):
    markers = {
        0: ArucoMarker(0, 0.165),
        4: ArucoMarker(4, 1.0, static=True),
    }
    original = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_100, markers=markers)
    path = tmp_path / "marker_set.toml"
    original.to_toml(path)
    loaded = ArucoMarkerSet.from_toml(path)
    assert loaded.markers[0].static is False
    assert loaded.markers[4].static is True


def test_pre_branch3_toml_loads(tmp_path):
    """A TOML file without static fields loads with static=False."""
    path = tmp_path / "old.toml"
    path.write_text("dictionary = 0\n\n[[markers]]\nid = 0\nsize_m = 0.165\n")
    loaded = ArucoMarkerSet.from_toml(path)
    assert loaded.markers[0].static is False


# -- MarkerLink --


def test_marker_link_construction():
    link = MarkerLink(marker_a=0, marker_b=8, corner_map=(1, 0, 3, 2), separation_m=0.004)
    assert link.marker_a == 0
    assert link.separation_m == 0.004


def test_marker_link_rejects_bad_corner_map():
    with pytest.raises(ValueError, match="permutation"):
        MarkerLink(marker_a=0, marker_b=1, corner_map=(0, 0, 1, 2))


def test_marker_link_rejects_negative_separation():
    with pytest.raises(ValueError, match="separation_m"):
        MarkerLink(marker_a=0, marker_b=1, corner_map=(0, 1, 2, 3), separation_m=-0.01)


def test_marker_set_with_links(tmp_path):
    markers = {0: ArucoMarker(0, 0.165), 8: ArucoMarker(8, 0.165)}
    link = MarkerLink(marker_a=0, marker_b=8, corner_map=(1, 0, 3, 2), separation_m=0.004)
    ms = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_100, markers=markers, links=(link,))
    path = tmp_path / "linked.toml"
    ms.to_toml(path)
    loaded = ArucoMarkerSet.from_toml(path)
    assert len(loaded.links) == 1
    assert loaded.links[0].corner_map == (1, 0, 3, 2)
    assert loaded.links[0].separation_m == 0.004


def test_marker_set_rejects_link_unknown_marker():
    markers = {0: ArucoMarker(0, 0.165)}
    link = MarkerLink(marker_a=0, marker_b=99, corner_map=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="unknown marker_b"):
        ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_100, markers=markers, links=(link,))


# -- ImagePoints.filter_to_objects --


def test_filter_to_objects():
    import pandas as pd
    from caliscope.core.point_data import ImagePoints

    df = pd.DataFrame(
        {
            "sync_index": [0, 0, 0, 0, 1, 1],
            "cam_id": [0, 0, 0, 0, 0, 0],
            "object_id": [0, 0, 37, 37, 0, 44],
            "keypoint_id": [0, 1, 0, 1, 0, 0],
            "img_loc_x": [100.0, 200.0, 300.0, 400.0, 110.0, 500.0],
            "img_loc_y": [100.0, 200.0, 300.0, 400.0, 110.0, 500.0],
        }
    )
    ip = ImagePoints(df)
    filtered = ip.filter_to_objects({0})
    assert len(filtered.df) == 3
    assert set(filtered.df["object_id"].unique()) == {0}


def test_generate_marker_image():
    ms = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_100,
        markers={0: ArucoMarker(0, 0.05)},
    )
    img = ms.generate_marker_image(0, 200)
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_generate_marker_image_unknown_id():
    ms = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_100,
        markers={0: ArucoMarker(0, 0.05)},
    )
    with pytest.raises(KeyError):
        ms.generate_marker_image(99, 200)


# -- ArucoTracker with ArucoMarkerSet --


def test_tracker_single_marker_with_marker_set():
    """Single-marker set works the same as old ArucoTarget."""
    from caliscope.trackers.aruco_tracker import ArucoTracker

    marker_set = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_100,
        markers={0: ArucoMarker(0, 0.05)},
    )
    tracker = ArucoTracker(marker_set=marker_set)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)
    img = np.ones((400, 400), dtype=np.uint8) * 255
    img[100:300, 100:300] = marker_img

    packet = tracker.get_points(img)
    assert packet.obj_loc is not None
    assert len(packet.obj_loc) == 4
    assert np.max(np.abs(packet.obj_loc[:, :2])) == pytest.approx(0.025, abs=1e-6)


def test_tracker_without_marker_set():
    """Without marker_set, obj_loc is None."""
    from caliscope.trackers.aruco_tracker import ArucoTracker

    tracker = ArucoTracker()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)
    img = np.ones((400, 400), dtype=np.uint8) * 255
    img[100:300, 100:300] = marker_img

    packet = tracker.get_points(img)
    assert packet.obj_loc is None
    assert len(packet.img_loc) == 4


def test_tracker_multi_marker_uses_per_marker_size():
    """Two markers of different sizes produce different obj_loc corners."""
    from caliscope.trackers.aruco_tracker import ArucoTracker

    small = ArucoMarker(0, 0.05)
    large = ArucoMarker(3, 0.10)
    marker_set = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_100,
        markers={0: small, 3: large},
    )

    img = np.ones((600, 800), dtype=np.uint8) * 255
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    marker_0_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 100)
    marker_3_img = cv2.aruco.generateImageMarker(aruco_dict, 3, 100)
    img[100:200, 100:200] = marker_0_img
    img[100:200, 400:500] = marker_3_img

    tracker = ArucoTracker(marker_set=marker_set)
    packet = tracker.get_points(img)

    assert packet.obj_loc is not None
    assert len(packet.obj_loc) > 0

    mask_0 = packet.object_id == 0
    mask_3 = packet.object_id == 3
    assert mask_0.any(), "Marker 0 not detected"
    assert mask_3.any(), "Marker 3 not detected"

    corners_0 = packet.obj_loc[mask_0]
    corners_3 = packet.obj_loc[mask_3]
    max_0 = np.max(np.abs(corners_0[:, :2]))
    max_3 = np.max(np.abs(corners_3[:, :2]))
    assert max_0 == pytest.approx(0.025, abs=1e-6)
    assert max_3 == pytest.approx(0.05, abs=1e-6)


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    ms = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_100,
        markers={
            0: ArucoMarker(0, 0.165),
            3: ArucoMarker(3, 0.10),
            7: ArucoMarker(7, 0.165),
        },
    )
    toml_path = debug_dir / "aruco_marker_set.toml"
    ms.to_toml(toml_path)
    loaded = ArucoMarkerSet.from_toml(toml_path)
    print(f"Round-trip OK: {len(loaded.markers)} markers")

    for mid in ms.markers:
        img = ms.generate_marker_image(mid, 200)
        cv2.imwrite(str(debug_dir / f"marker_{mid}.png"), img)
        print(f"Saved marker_{mid}.png ({img.shape})")
