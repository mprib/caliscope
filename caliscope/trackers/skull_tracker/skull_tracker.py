import logging
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points

logger = logging.getLogger(__name__)


class SkullTracker(Tracker):
    """
    Queue-based skull tracker using MediaPipe Holistic.
    Provides CONSTANT 3D skull-centric coordinates for landmarks (like a Charuco board).
    The obj_loc values are the same across all frames and cameras.
    """

    def __init__(self) -> None:
        # Queue-based processing infrastructure
        self.in_queues: dict[int, Queue] = {}
        self.out_queues: dict[int, Queue] = {}
        self.threads: dict[int, Thread] = {}

        # Load canonical model and configuration
        self.canonical_vertices = self._load_canonical_vertices()
        self.skull_landmark_ids, self.weights = self._load_procrustes_basis()

        # Compute CONSTANT 3D positions for all landmarks in skull frame (in METERS)
        self.skull_landmark_positions = self._compute_skull_landmark_positions()

        # Minimum threshold for detection
        self.min_landmarks_threshold = int(len(self.skull_landmark_ids) * 0.7)

        logger.info(f"SkullTracker initialized with {len(self.skull_landmark_ids)} landmarks")
        logger.info("Skull frame origin: nose tip, scale: ~0.065m inter-eye distance")

    def _load_canonical_vertices(self) -> NDArray[np.float64]:
        """Load 3D vertex positions from canonical_face_model.obj"""
        vertices = []
        obj_path = Path(__file__).parent / "canonical_face_model.obj"

        with open(obj_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

        vertices_array = np.array(vertices, dtype=np.float64)
        if vertices_array.shape[0] != 468:
            raise ValueError(f"Expected 468 vertices, got {vertices_array.shape[0]}")

        return vertices_array

    def _load_procrustes_basis(self) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        """Parse pbtxt to get landmark IDs and weights for Procrustes alignment"""
        import re

        pbtxt_path = Path(__file__).parent / "geometry_pipeline_metadata_landmarks.pbtxt"
        landmark_ids = []
        weights = []

        with open(pbtxt_path, "r") as f:
            for line in f:
                # FIX: Handle same-line landmark_id and weight
                if "landmark_id:" in line:
                    match = re.search(r"landmark_id:\s*(\d+)", line)
                    if match:
                        landmark_ids.append(int(match.group(1)))
                if "weight:" in line:  # Use 'if' not 'elif' - they can be on same line
                    match = re.search(r"weight:\s*([\d.]+)", line)
                    if match:
                        weights.append(float(match.group(1)))

        if len(landmark_ids) != len(weights):
            logger.warning(f"Mismatch: {len(landmark_ids)} landmark_ids vs {len(weights)} weights")
            # Take the minimum length to avoid alignment issues
            min_len = min(len(landmark_ids), len(weights))
            landmark_ids = landmark_ids[:min_len]
            weights = weights[:min_len]

        return np.array(landmark_ids, dtype=np.int32), np.array(weights, dtype=np.float64)

    def _compute_skull_landmark_positions(self) -> NDArray[np.float64]:
        """
        Compute CONSTANT 3D positions for all landmarks in skull-centric frame.
        These values are the SAME across all frames and cameras.

        Skull frame definition:
        - Origin: Nose tip (landmark 1)
        - X-axis: Right eye (362) - Left eye (133) → points right
        - Z-axis: Forward from face
        - Y-axis: Up (right-handed coordinate system)
        - Units: Meters
        """
        # Scale factor: canonical units → meters (inter-eye distance = 0.065m)
        scale_to_meters = 0.065 / np.linalg.norm(self.canonical_vertices[133] - self.canonical_vertices[362])

        # Scale all vertices to meters
        scaled_vertices = self.canonical_vertices * scale_to_meters

        # Define skull frame axes from SCALED geometry
        nose_tip = scaled_vertices[1]
        left_eye = scaled_vertices[133]
        right_eye = scaled_vertices[362]

        # X axis: eye line (right - left)
        x_axis = right_eye - left_eye
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Z axis: forward direction (from nose to eye plane)
        eye_center = (left_eye + right_eye) / 2
        z_axis = eye_center - nose_tip
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Y axis: up (cross product of Z × X)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Build coordinate system matrix (rotation only)
        coordinate_system = np.column_stack([x_axis, y_axis, z_axis])

        # Transform all vertices: V_skull = R^T @ (V_scaled - nose_tip)
        # This puts nose tip at origin and aligns axes
        centered_vertices = scaled_vertices - nose_tip
        skull_positions = (coordinate_system.T @ centered_vertices.T).T

        # Debug: Log some key landmark positions
        if not hasattr(self, "_logged_positions"):
            self._logged_positions = True
            logger.info("Skull landmark positions (meters):")
            logger.info(f"  Nose tip (1): {skull_positions[1]}")
            logger.info(f"  Left eye (133): {skull_positions[133]}")
            logger.info(f"  Right eye (362): {skull_positions[362]}")
            logger.info(f"  Inter-eye distance: {np.linalg.norm(skull_positions[133] - skull_positions[362]):.3f}m")

        return skull_positions

    def run_frame_processor(self, port: int, rotation_count: int):
        """
        Background thread processor for each camera port.
        Only detects landmarks and returns 2D positions + CONSTANT 3D positions.
        """
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                frame = self.in_queues[port].get()

                if frame is None:  # Shutdown signal
                    break

                # Apply rotation as needed
                rotated_frame = apply_rotation(frame, rotation_count)
                height, width, _ = rotated_frame.shape

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)

                # Run holistic detection
                results = holistic.process(rgb_frame)

                # Initialize empty packet
                point_packet = PointPacket(
                    point_id=np.array([], dtype=np.int32),
                    img_loc=np.empty((0, 2), dtype=np.float64),
                    obj_loc=np.empty((0, 3), dtype=np.float64),
                )

                # Check if face landmarks are detected
                if results.face_landmarks:
                    # Extract all 468 face landmarks from holistic
                    face_landmarks = results.face_landmarks

                    # Build arrays for all landmarks
                    point_ids = np.arange(468, dtype=np.int32)
                    img_locs = np.array(
                        [[landmark.x * width, landmark.y * height] for landmark in face_landmarks.landmark],
                        dtype=np.float64,
                    )

                    # Check how many skull-rigid landmarks are actually visible
                    skull_mask = np.isin(point_ids, self.skull_landmark_ids)
                    visible_skull_count = np.sum(skull_mask)

                    if visible_skull_count >= self.min_landmarks_threshold:
                        # Use PRE-COMPUTED constant 3D positions (THE KEY FIX)
                        filtered_ids = point_ids[skull_mask]
                        filtered_img = img_locs[skull_mask]
                        filtered_obj = self.skull_landmark_positions[filtered_ids]

                        # Unrotate image points back to original orientation
                        filtered_img = unrotate_points(filtered_img, rotation_count, width, height)

                        point_packet = PointPacket(point_id=filtered_ids, img_loc=filtered_img, obj_loc=filtered_obj)

                        logger.debug(f"Port {port}: Detected {visible_skull_count} skull landmarks")
                    else:
                        logger.debug(
                            f"Port {port}: Only {visible_skull_count} skull landmarks visible "
                            f"(need {self.min_landmarks_threshold})"
                        )

                # Put result in output queue
                self.out_queues[port].put(point_packet)

    def get_points(self, frame: NDArray[np.uint8], port: int, rotation_count: int) -> PointPacket:
        """
        Main interface called by SynchronizedStreamManager.
        Queues frame for processing and returns result.
        """
        # Initialize queues and thread for this port if not exists
        if port not in self.in_queues:
            self.in_queues[port] = Queue(maxsize=1)
            self.out_queues[port] = Queue(maxsize=1)

            self.threads[port] = Thread(
                target=self.run_frame_processor,
                args=(port, rotation_count),
                daemon=True,
                name=f"SkullTracker_Port_{port}",
            )
            self.threads[port].start()

        # Put frame in input queue
        self.in_queues[port].put(frame)

        # Get result from output queue
        point_packet = self.out_queues[port].get()

        return point_packet

    @property
    def name(self) -> str:
        return "SKULL"

    def get_point_name(self, point_id: int) -> str:
        return f"landmark_{point_id}"

    def scatter_draw_instructions(self, point_id: int) -> dict[str, Any]:
        return {"radius": 3, "color": (0, 220, 0), "thickness": 2}

    def get_connected_points(self) -> set[tuple[int, int]]:
        return set()
