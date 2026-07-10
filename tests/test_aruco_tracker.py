import logging
import pytest
import cv2
import numpy as np

from caliscope import __root__
from caliscope.trackers.aruco_tracker import ArucoTracker

logger = logging.getLogger(__name__)


def test_aruco_tracker_instantiation():
    """Test that ArucoTracker can be instantiated with default parameters."""
    tracker = ArucoTracker()
    assert tracker.name == "ARUCO"
    assert tracker.inverted is False
    assert tracker.dictionary == cv2.aruco.DICT_4X4_100


def test_aruco_tracker_detection():
    """Test marker detection on a sample frame from test fixture."""
    # Load test video frame
    fixture_dir = __root__ / "tests/sessions/post_optimization"
    sample_video_path = fixture_dir / "calibration/extrinsic/cam_0.mp4"

    capture = cv2.VideoCapture(str(sample_video_path))
    success, frame = capture.read()
    capture.release()

    assert success, "Failed to load test frame"

    # Create tracker with inverted=True for test data
    tracker = ArucoTracker(inverted=True)

    # Process frame
    point_packet = tracker.get_points(frame)

    # Validate PointPacket structure
    assert point_packet is not None
    assert isinstance(point_packet.object_id, np.ndarray)
    assert isinstance(point_packet.keypoint_id, np.ndarray)
    assert isinstance(point_packet.img_loc, np.ndarray)
    assert point_packet.obj_loc is None  # ArUco has no board reference

    # Should detect markers
    assert len(point_packet.keypoint_id) > 0
    assert len(point_packet.img_loc) == len(point_packet.keypoint_id)
    assert point_packet.img_loc.shape[1] == 2  # x,y coordinates

    logger.info(
        f"Detected {len(point_packet.keypoint_id)} points across {len(np.unique(point_packet.object_id))} markers"
    )


def test_aruco_get_point_name():
    """Test minimal point name implementation."""
    tracker = ArucoTracker()

    # Test various point IDs
    assert tracker.get_point_name(121) == "121"
    assert tracker.get_point_name(42) == "42"
    assert tracker.get_point_name(0) == "0"


def test_aruco_draw_instructions():
    """Test drawing instructions return correct format."""
    tracker = ArucoTracker()

    instructions = tracker.scatter_draw_instructions(keypoint_id=123)

    assert isinstance(instructions, dict)
    assert "radius" in instructions
    assert "color" in instructions
    assert "thickness" in instructions

    # Verify green color scheme (BGR format)
    assert instructions["radius"] == 4
    assert instructions["color"] == (0, 255, 0)  # Green in BGR
    assert instructions["thickness"] == -1  # Filled circle


def test_aruco_point_id_mapping():
    """Verify object_id is the marker_id and keypoint_id is the corner_index (0-3)."""
    tracker = ArucoTracker(inverted=True)
    fixture_dir = __root__ / "tests/sessions/post_optimization"
    sample_video_path = fixture_dir / "calibration/extrinsic/cam_0.mp4"

    capture = cv2.VideoCapture(str(sample_video_path))
    success, frame = capture.read()
    capture.release()

    point_packet = tracker.get_points(frame)

    # Check that each marker has exactly the expected corner indices
    unique_marker_ids = np.unique(point_packet.object_id)

    for marker_id in unique_marker_ids:
        corner_ids = point_packet.keypoint_id[point_packet.object_id == marker_id]
        expected_ids = np.array([0, 1, 2, 3])

        np.testing.assert_array_equal(
            np.sort(corner_ids),
            np.sort(expected_ids),
            err_msg=f"Marker {marker_id} doesn't have expected corner IDs {expected_ids}",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_aruco_tracker_instantiation()
    # test_aruco_tracker_detection()
    # test_aruco_point_id_mapping()
    # test_aruco_get_point_name()
    # test_aruco_draw_instructions()
    pytest.main(["tests/test_aruco_tracker.py"])
