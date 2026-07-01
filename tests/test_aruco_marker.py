import numpy as np
import pytest
import cv2

from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet


# -- ArucoMarker --


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
