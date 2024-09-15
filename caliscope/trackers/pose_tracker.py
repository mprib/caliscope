from queue import Queue
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np

import caliscope.logger

# cap = cv2.VideoCapture(0)
from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points

logger = caliscope.logger.get(__name__)

POINT_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


class PoseTracker(Tracker):
    def __init__(self) -> None:
        # each port gets its own mediapipe context manager
        # use a dictionary of queues for passing
        self.in_queues = {}
        self.out_queues = {}
        self.threads = {}

    @property
    def name(self):
        return "POSE"

    def run_frame_processor(self, port: int, rotation_count: int):
        # Create a MediaPipe pose instance
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        ) as pose:
            while True:
                frame = self.in_queues[port].get()
                # apply rotation as needed
                frame = apply_rotation(frame, rotation_count)

                height, width, color = frame.shape
                # Convert the image to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)

                # initialize variables so none will be created if no points detected
                point_ids = []
                landmark_xy = []

                if results.pose_landmarks:
                    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                        point_ids.append(landmark_id)

                        # mediapipe expresses in terms of percent of frame, so must map to pixel position
                        x, y = int(landmark.x * width), int(landmark.y * height)
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

    def get_point_name(self, point_id) -> str:
        return POINT_NAMES[point_id]

    def scatter_draw_instructions(self, point_id: int) -> dict:
        if self.get_point_name(point_id).startswith("left"):
            rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        elif self.get_point_name(point_id).startswith("right"):
            rules = {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        else:
            rules = {"radius": 5, "color": (220, 0, 220), "thickness": 3}

        return rules
