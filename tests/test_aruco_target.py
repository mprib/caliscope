"""
ArUco target domain tests: dataclass, persistence, tracker enhancement.

Test data lives in tests/sessions/aruco_extrinsic/ as extracted JPGs from
the chessboard_aruco project's extrinsic videos. This avoids storing large
video files in the repo.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest

from caliscope.core.aruco_target import ArucoTarget
from caliscope.persistence import PersistenceError, load_aruco_target, save_aruco_target
from caliscope.trackers.aruco_tracker import ArucoTracker


# -- Dataclass ---------------------------------------------------------------


def test_aruco_target_immutable():
    """Frozen dataclass should reject mutation."""
    target = ArucoTarget.single_marker()
    with pytest.raises(FrozenInstanceError):
        target.marker_size_m = 0.1  # type: ignore[misc]


def test_single_marker_factory_defaults():
    """single_marker() creates valid target with default params."""
    target = ArucoTarget.single_marker()
    assert target.marker_ids == [0]
    assert target.dictionary == cv2.aruco.DICT_4X4_100
    assert target.marker_size_m == 0.05


def test_corner_positions_centered_at_origin():
    """Corners should be symmetric around origin."""
    target = ArucoTarget.single_marker(marker_size_m=0.1)
    corners = target.get_corner_positions(0)

    centroid = corners.mean(axis=0)
    np.testing.assert_array_almost_equal(centroid, [0, 0, 0])


def test_corner_positions_correct_order():
    """Corners follow OpenCV ArUco convention: TL, TR, BR, BL with Y-up."""
    target = ArucoTarget.single_marker(marker_size_m=0.1)
    corners = target.get_corner_positions(0)

    s = 0.05  # half-size
    # OpenCV ArUco: origin at center, X right, Y up
    expected = np.array([[-s, +s, 0], [+s, +s, 0], [+s, -s, 0], [-s, -s, 0]])
    np.testing.assert_array_almost_equal(corners, expected)


def test_marker_image_generation():
    """generate_marker_image produces annotated BGR image."""
    target = ArucoTarget.single_marker()
    img = target.generate_marker_image(0)

    assert img.ndim == 3  # BGR
    assert img.shape[2] == 3
    assert img.dtype == np.uint8
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_marker_image_invalid_id():
    """generate_marker_image rejects unknown marker ID."""
    target = ArucoTarget.single_marker(marker_id=0)
    with pytest.raises(KeyError, match="Marker 99"):
        target.generate_marker_image(99)


# -- Persistence -------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path):
    """Save/load round-trip preserves all fields."""
    original = ArucoTarget.single_marker(marker_id=7, marker_size_m=0.08)
    file_path = tmp_path / "aruco_target.toml"

    save_aruco_target(original, file_path)
    loaded = load_aruco_target(file_path)

    assert loaded.dictionary == original.dictionary
    assert loaded.marker_size_m == original.marker_size_m
    assert loaded.marker_ids == original.marker_ids
    np.testing.assert_array_almost_equal(loaded.corners[7], original.corners[7])


def test_load_nonexistent_file(tmp_path: Path):
    """Load from nonexistent path raises PersistenceError."""
    with pytest.raises(PersistenceError, match="not found"):
        load_aruco_target(tmp_path / "nonexistent.toml")


def test_save_creates_parent_directory(tmp_path: Path):
    """Save should create parent directories if they don't exist."""
    original = ArucoTarget.single_marker()
    file_path = tmp_path / "nested" / "path" / "aruco_target.toml"
    save_aruco_target(original, file_path)

    assert file_path.exists()
    loaded = load_aruco_target(file_path)
    assert loaded.marker_ids == original.marker_ids


# -- Tracker enhancement (real frames) ---------------------------------------

TEST_DATA_DIR = Path("tests/sessions/aruco_extrinsic")


def test_tracker_with_target_populates_obj_loc():
    """When aruco_target provided, obj_loc should be populated."""
    frame_path = TEST_DATA_DIR / "cam_0_frame_0050.jpg"
    if not frame_path.exists():
        pytest.skip(f"Test data not extracted: {frame_path}")

    target = ArucoTarget.single_marker(marker_id=0, marker_size_m=0.05)
    tracker = ArucoTracker(aruco_target=target)

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    packet = tracker.get_points(frame)

    assert packet.obj_loc is not None
    assert packet.obj_loc.shape == (4, 3)  # 4 corners, 3D
    assert len(packet.point_id) == 4


def test_tracker_without_target_has_none_obj_loc():
    """Backward compat: without aruco_target, obj_loc=None."""
    frame_path = TEST_DATA_DIR / "cam_0_frame_0050.jpg"
    if not frame_path.exists():
        pytest.skip(f"Test data not extracted: {frame_path}")

    tracker = ArucoTracker()  # No target

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    packet = tracker.get_points(frame)

    assert packet.obj_loc is None
    assert len(packet.point_id) > 0  # Still detects markers


def test_tracker_filters_to_target_marker_ids():
    """Tracker only reports markers in target.marker_ids."""
    target = ArucoTarget.single_marker(marker_id=0)
    tracker = ArucoTracker(aruco_target=target)

    frame_path = TEST_DATA_DIR / "cam_0_frame_0050.jpg"
    if not frame_path.exists():
        pytest.skip(f"Test data not extracted: {frame_path}")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    packet = tracker.get_points(frame)

    # All point IDs should be for marker 0 (0-9 range)
    for pid in packet.point_id:
        assert pid // 10 == 0


def test_obj_loc_matches_target_corners():
    """obj_loc values should match target.corners positions."""
    target = ArucoTarget.single_marker(marker_id=0, marker_size_m=0.05)
    tracker = ArucoTracker(aruco_target=target)

    frame_path = TEST_DATA_DIR / "cam_0_frame_0050.jpg"
    if not frame_path.exists():
        pytest.skip(f"Test data not extracted: {frame_path}")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    packet = tracker.get_points(frame)

    expected_corners = target.corners[0]

    for i, pid in enumerate(packet.point_id):
        corner_index = pid % 10  # 0-3, matches OpenCV corner order
        np.testing.assert_array_almost_equal(packet.obj_loc[i], expected_corners[corner_index])


def test_no_marker_returns_empty_packet_with_target():
    """Empty frame with target provided returns empty obj_loc array."""
    frame_path = TEST_DATA_DIR / "cam_0_frame_0000.jpg"
    if not frame_path.exists():
        pytest.skip(f"Test data not extracted: {frame_path}")

    target = ArucoTarget.single_marker()
    tracker = ArucoTracker(aruco_target=target)

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    packet = tracker.get_points(frame)

    assert len(packet.point_id) == 0
    assert packet.obj_loc is not None  # Not None when target is set
    assert packet.obj_loc.shape == (0, 3)


def test_cross_camera_detection():
    """All 4 cameras detect marker 0 on a synchronized frame."""
    frame_paths = [TEST_DATA_DIR / f"cam_{i}_frame_0200.jpg" for i in range(4)]

    available_frames = [(p, cv2.imread(str(p))) for p in frame_paths if p.exists()]
    if len(available_frames) < 2:
        pytest.skip("Need at least 2 camera frames for cross-camera test")

    target = ArucoTarget.single_marker(marker_id=0)
    tracker = ArucoTracker(aruco_target=target)

    packets = []
    for path, frame in available_frames:
        if frame is None:
            continue
        packet = tracker.get_points(frame)
        if len(packet.point_id) > 0:
            packets.append((path, packet))

    assert len(packets) >= 2, "Need at least 2 successful detections"

    # All should have 4 corners with obj_loc
    for path, packet in packets:
        assert len(packet.point_id) == 4, f"Expected 4 corners from {path}"
        assert packet.obj_loc is not None
        assert packet.obj_loc.shape == (4, 3)

    # All cameras should report the same point IDs (same marker)
    reference_ids = sorted(packets[0][1].point_id.tolist())
    for path, packet in packets[1:]:
        assert sorted(packet.point_id.tolist()) == reference_ids, f"Point ID mismatch: {packets[0][0]} vs {path}"


# -- Debug harness -----------------------------------------------------------

if __name__ == "__main__":
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Test domain object
    target = ArucoTarget.single_marker(marker_id=0, marker_size_m=0.05)
    print(f"Created target with markers: {target.marker_ids}")
    print(f"Corner positions:\n{target.corners[0]}")

    # Generate and save marker image
    marker_img = target.generate_marker_image(0)
    cv2.imwrite(str(debug_dir / "marker_0.png"), marker_img)
    print(f"Saved marker image to {debug_dir}/marker_0.png")

    # Test persistence round-trip
    toml_path = debug_dir / "aruco_target.toml"
    save_aruco_target(target, toml_path)
    loaded = load_aruco_target(toml_path)
    print(f"\nPersistence round-trip OK: {loaded.marker_ids}, size={loaded.marker_size_m}m")

    # Test tracker with target
    frame_path = TEST_DATA_DIR / "cam_0_frame_0050.jpg"
    if frame_path.exists():
        tracker = ArucoTracker(aruco_target=target)
        frame = cv2.imread(str(frame_path))
        packet = tracker.get_points(frame)

        print(f"\nDetected {len(packet.point_id)} points")
        print(f"Point IDs: {packet.point_id}")
        print(f"obj_loc shape: {packet.obj_loc.shape if packet.obj_loc is not None else None}")

        # Draw detection
        annotated = frame.copy()
        for pid, img_loc, obj_loc in zip(packet.point_id, packet.img_loc, packet.obj_loc):
            x, y = int(img_loc[0]), int(img_loc[1])
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), 2)
            label = f"{pid}: ({obj_loc[0]:.3f}, {obj_loc[1]:.3f})"
            cv2.putText(annotated, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        cv2.imwrite(str(debug_dir / "aruco_detection.jpg"), annotated)
        print(f"Saved annotated frame to {debug_dir}/aruco_detection.jpg")
    else:
        print(f"\nTest frame not found: {frame_path}")
        print("Extract test frames first.")
