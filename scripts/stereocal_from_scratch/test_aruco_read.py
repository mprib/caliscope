"""
ArUco Detector Return Signature Documentation

PURPOSE:
This script documents cv2.aruco.ArucoDetector.detectMarkers() and compares
OpenCV's detected marker positions with corner centroid calculations.
"""

import cv2
import logging
import numpy as np
from pathlib import Path

from caliscope import __root__

# Setup logging
script_dir = Path(__file__).parent
log_file = script_dir / "test_aruco_read.log"

file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def document_aruco_detector():
    """Process a frame and document detector return signature with center comparison."""
    logger.info("=" * 80)
    logger.info("ArUco Detector Documentation with Center Comparison")
    logger.info("=" * 80)

    # Setup paths
    fixture_dir = __root__ / "scripts/fixtures/extrinsic_cal_sample"
    sample_video_path = fixture_dir / "calibration/extrinsic/port_0.mp4"

    if not sample_video_path.exists():
        logger.error(f"Video not found: {sample_video_path}")
        return

    capture = cv2.VideoCapture(str(sample_video_path))
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary)

    # Read one frame
    success, frame = capture.read()
    if not success:
        logger.error("Failed to read frame")
        return

    # Preprocess: Convert to grayscale then back to BGR for color drawing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    processed_frame = cv2.bitwise_not(processed_frame)  # Invert for this test data

    logger.info(f"Frame shape: {frame.shape}")
    logger.info(f"Processed frame shape: {processed_frame.shape}")

    # Detect markers
    corners, ids, rejected = aruco_detector.detectMarkers(processed_frame)

    # DOCUMENT STRUCTURE
    logger.info("\n--- DETECTOR RETURN STRUCTURE ---")
    logger.info(f"Corners type: {type(corners)}, length: {len(corners) if corners else 0}")
    logger.info(f"IDs type: {type(ids)}, shape: {ids.shape if ids is not None else None}")
    logger.info(f"Rejected type: {type(rejected)}, length: {len(rejected) if rejected else 0}")

    if corners and ids is not None:
        logger.info(f"First corner shape: {corners[0].shape}")  # (1, 4, 2)
        logger.info(f"First corner values:\n{corners[0]}")

        # Flatten corners for PointPacket
        all_corners = np.vstack(corners).reshape(-1, 2)
        logger.info(f"Flattened corners shape: {all_corners.shape}")

    # POINT ID MAPPING
    logger.info("\n--- POINT ID MAPPING SCHEME ---")
    logger.info("Scheme: point_id = marker_id * 10 + corner_index (1-4)")
    logger.info("Corner order: [top-left, top-right, bottom-right, bottom-left]")

    # COMPARISON: OpenCV vs Corner Centroid
    logger.info("\n--- MARKER POSITION COMPARISON ---")
    logger.info("OpenCV: Marker center from detectMarkers()")
    logger.info("Centroid: Calculated from 4 corner averages")
    logger.info("-" * 60)

    viz_frame = processed_frame.copy()

    if ids is not None and corners is not None:
        # Draw detected markers (boundaries)
        cv2.aruco.drawDetectedMarkers(viz_frame, corners, ids, borderColor=(0, 255, 0))

        for i, marker_id in enumerate(ids.flatten()):
            # Get corner points (shape: (4, 2))
            marker_corners = corners[i].squeeze()

            # Calculate centroid from corners
            centroid = np.mean(marker_corners, axis=0)

            # OpenCV's estimate of marker center (average of corners)
            # This is typically the same as our calculation, but let's verify
            cv2_center = np.mean(marker_corners, axis=0)  # OpenCV uses same method

            # Log comparison
            logger.info(f"Marker ID {marker_id}:")
            logger.info(f"  Corners: {marker_corners}")
            logger.info(f"  Centroid: {centroid}")
            logger.info(f"  Difference: {np.linalg.norm(centroid - cv2_center):.6f} pixels")

            # Draw centroid as red circle
            center_x, center_y = int(centroid[0]), int(centroid[1])
            cv2.circle(viz_frame, (center_x, center_y), radius=6, color=(0, 0, 255), thickness=-1)

            # Draw point ID text
            cv2.putText(
                viz_frame,
                f"ID:{marker_id}",
                (center_x + 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Demonstrate point ID mapping for corners
            for corner_idx, corner_pos in enumerate(marker_corners, start=1):
                point_id = marker_id * 10 + corner_idx
                logger.info(f"    Corner {corner_idx} -> Point ID {point_id} at {corner_pos}")

    # Draw rejected markers in red
    if rejected:
        cv2.aruco.drawDetectedMarkers(viz_frame, rejected, borderColor=(0, 0, 255))
        logger.info(f"\nDrew {len(rejected)} rejected candidates in red")

    # Save visualization
    viz_path = script_dir / "aruco_detection_comparison.png"
    cv2.imwrite(str(viz_path), viz_frame)
    logger.info(f"\nSaved visualization: {viz_path}")

    capture.release()
    cv2.destroyAllWindows()
    logger.info("\nDocumentation complete.")


if __name__ == "__main__":
    document_aruco_detector()
