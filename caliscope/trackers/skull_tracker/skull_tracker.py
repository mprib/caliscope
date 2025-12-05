import logging
from queue import Queue
from threading import Thread
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

# cap = cv2.VideoCapture(0)
from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points

logger = logging.getLogger(__name__)


###
class SkullTracker(Tracker):
    def __init__(self) -> None:
        # each port gets its own mediapipe context manager
        # use a dictionary of queues for passing
        self.in_queues: dict[int, Queue] = {}
        self.out_queues: dict[int, Queue] = {}
        self.threads: dict[int, Thread] = {}
        # wireframe_spec_path = Path(Path(__file__).parent, "skull_wireframe.toml")
        # self.wireframe = get_wireframe(wireframe_spec_path, POINT_NAMES)

    @property
    def name(self):
        return "SKULL"

    def run_frame_processor(self, port: int, rotation_count: int):
        # Create a MediaPipe pose instance
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
            while True:
                frame = self.in_queues[port].get()
                # apply rotation as needed
                frame = apply_rotation(frame, rotation_count)

                height, width, color = frame.shape
                # Convert the image to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)

                # initialize variables so none will be created if no points detected
                point_ids: list[int] = []
                landmark_xy: list[tuple[float, float]] = []

                if results.face_landmarks:
                    for landmark_id, landmark in enumerate(results.face_landmarks.landmark):
                        # mediapipe expresses in terms of percent of frame, so must map to pixel positionFSET
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        if landmark.x < 0 or landmark.x > 1 or landmark.y < 0 or landmark.y > 1:
                            # ignore
                            pass
                        else:
                            point_ids.append(landmark_id)
                            landmark_xy.append((x, y))

                point_ids = np.array(point_ids)
                landmark_xy = np.array(landmark_xy)
                landmark_xy = unrotate_points(landmark_xy, rotation_count, width, height)
                point_packet = PointPacket(point_ids, landmark_xy)

                self.out_queues[port].put(point_packet)

    def get_points(self, frame: np.ndarray, port: int, rotation_count: int) -> PointPacket:
        if port not in self.in_queues.keys():
            self.in_queues[port] = Queue(1)
            self.out_queues[port] = Queue(1)

            self.threads[port] = Thread(
                target=self.run_frame_processor,
                args=(port, rotation_count),
                daemon=True,
            )

            self.threads[port].start()

        self.in_queues[port].put(frame)
        point_packet = self.out_queues[port].get()

        return point_packet

    def get_point_name(self, point_id: int) -> str:
        # TODO map with name dict
        return str(point_id)

    def scatter_draw_instructions(self, point_id: int) -> dict[str, Any]:
        point_name = self.get_point_name(point_id)

        if point_name.startswith("left"):
            rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        elif point_name.startswith("right"):
            rules = {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        else:
            rules = {"radius": 1, "color": (220, 0, 220), "thickness": 1}

        return rules

    def get_connected_points(self) -> set[tuple[int, int]]:
        return super().get_connected_points()
