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
    fixture_dir = __root__ / "scripts/fixtures/extrinsic_cal_sample"
    sample_video_path = fixture_dir / "calibration/extrinsic/port_0.mp4"

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
    assert isinstance(point_packet.point_id, np.ndarray)
    assert isinstance(point_packet.img_loc, np.ndarray)
    assert point_packet.obj_loc is None  # ArUco has no board reference

    # Should detect markers
    assert len(point_packet.point_id) > 0
    assert len(point_packet.img_loc) == len(point_packet.point_id)
    assert point_packet.img_loc.shape[1] == 2  # x,y coordinates

    logger.info(
        f"Detected {len(point_packet.point_id)} points across {len(np.unique(point_packet.point_id // 10))} markers"
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

    instructions = tracker.scatter_draw_instructions(point_id=123)

    assert isinstance(instructions, dict)
    assert "radius" in instructions
    assert "color" in instructions
    assert "thickness" in instructions

    # Verify green color scheme (BGR format)
    assert instructions["radius"] == 4
    assert instructions["color"] == (0, 255, 0)  # Green in BGR
    assert instructions["thickness"] == -1  # Filled circle


def test_aruco_point_id_mapping():
    """Verify point ID scheme: marker_id * 10 + corner_index (1-4)."""
    tracker = ArucoTracker(inverted=True)
    fixture_dir = __root__ / "tests/sessions/post_optimization"
    sample_video_path = fixture_dir / "calibration/extrinsic/port_0.mp4"

    capture = cv2.VideoCapture(str(sample_video_path))
    success, frame = capture.read()
    capture.release()

    point_packet = tracker.get_points(frame)

    # Check that point IDs follow the expected pattern
    unique_marker_ids = np.unique(point_packet.point_id // 10)

    for marker_id in unique_marker_ids:
        corner_ids = point_packet.point_id[point_packet.point_id // 10 == marker_id]
        expected_ids = np.array([marker_id * 10 + i for i in range(1, 5)])

        np.testing.assert_array_equal(
            np.sort(corner_ids),
            np.sort(expected_ids),
            err_msg=f"Marker {marker_id} doesn't have expected corner IDs {expected_ids}",
        )


# NOTE: Test below corrupted by marker 8 which is symmetrical to it's rotated self.
# Reincorporate this test when you have footage with double sided ArUco flag.
# def test_aruco_mirror_detection():
#     """
#     Test that mirror detection finds markers in flipped frames and returns
#     coordinates in the flipped frame's coordinate space.
#     """
#     # Load a normal frame with markers (representing front-side view)
#     fixture_dir = __root__ / "tests/sessions/post_optimization"
#     sample_video_path = fixture_dir / "calibration/extrinsic/port_0.mp4"
#
#     capture = cv2.VideoCapture(str(sample_video_path))
#     success, original_frame = capture.read()
#     capture.release()
#
#     assert success, "Failed to load test frame"
#
#     # Flip frame horizontally to simulate viewing marker from back side
#     flipped_frame = cv2.flip(original_frame, 1)
#     frame_height, frame_width = flipped_frame.shape[:2]
#
#     # snapshot for ad hoc inspection
#     if __name__ == "__main__":
#         cv2.imwrite("original_image.png", original_frame)
#         cv2.imwrite("mirror image.png", flipped_frame)
#
#     # Create tracker with mirror search enabled
#     # test footage happens to be inverted
#     tracker = ArucoTracker(inverted=True, mirror_flag_search=True)
#
#     # Process original frame - should detect markers via mirror search
#     point_packet = tracker.get_points(original_frame, port=0, rotation_count=0)
#     flipped_point_packet = tracker.get_points(flipped_frame, port=0, rotation_count=0)
#
#     for packet in [point_packet, flipped_point_packet]:
#         # Should detect markers
#         assert len(point_packet.point_id) > 0, "Mirror detection should find markers in flipped frame"
#
#         # Verify coordinates are within flipped frame bounds
#         assert np.all(point_packet.img_loc[:, 0] >= 0), "All x coordinates should be >= 0"
#         assert np.all(point_packet.img_loc[:, 0] <= frame_width), "All x coordinates should be <= frame_width"
#         assert np.all(point_packet.img_loc[:, 1] >= 0), "All y coordinates should be >= 0"
#         assert np.all(point_packet.img_loc[:, 1] <= frame_height), "All y coordinates should be <= frame_height"
#
#     # Both should detect the same markers (same IDs)
#     np.testing.assert_array_equal(
#         np.sort(point_packet.point_id),
#         np.sort(flipped_point_packet.point_id),
#         err_msg="Mirror detection should find same markers as normal detection",
#     )
#
#     point_packet = None
#
#     # But coordinates should be different (mirrored)
#     # For each marker, verify x coordinates are mirrored and y coordinates are preserved
#     unique_marker_ids = np.unique(point_packet.point_id // 10)
#
#     for marker_id in unique_marker_ids:
#         # Get corners for this marker from both detections
#         orig_mask = point_packet.point_id // 10 == marker_id
#         flipped_mask = flipped_point_packet.point_id // 10 == marker_id
#
#         orig_corners = point_packet.img_loc[orig_mask]
#         flipped_corners = flipped_point_packet.img_loc[flipped_mask]
#
#         # Sort by corner index to ensure alignment
#         orig_corner_ids = point_packet.point_id[orig_mask] % 10
#         flipped_corner_ids = flipped_point_packet.point_id[flipped_mask] % 10
#
#         orig_order = np.argsort(orig_corner_ids)
#         flipped_order = np.argsort(flipped_corner_ids)
#
#         orig_corners = orig_corners[orig_order]
#         flipped_corners = flipped_corners[flipped_order]
#
#         # y coordinates should be preserved (vertical position doesn't change with horizontal flip)
#         np.testing.assert_allclose(
#             orig_corners[:, 1],
#             flipped_corners[:, 1],
#             rtol=1e-5,
#             err_msg="Y coordinates should be preserved during mirror detection",
#         )
#
#         # x coordinates should be mirrored: flipped_x = frame_width - original_x
#         expected_flipped_x = frame_width - orig_corners[:, 0]
#         np.testing.assert_allclose(
#             flipped_corners[:, 0],
#             expected_flipped_x,
#             rtol=1e-5,
#             err_msg="X coordinates should be correctly mirrored",
#         )
#
#     logger.info(f"Mirror detection test passed: {len(unique_marker_ids)} markers correctly detected and transformed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_aruco_tracker_instantiation()
    # test_aruco_tracker_detection()
    # test_aruco_point_id_mapping()
    # test_aruco_get_point_name()
    # test_aruco_draw_instructions()
    pytest.main(["tests/test_aruco_tracker.py"])
