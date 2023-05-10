import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)
from threading import Thread, Event
from queue import Queue

import mediapipe as mp
import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
from pyxy3d.interface import Tracker, PointPacket

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
        self.in_queue = Queue(-1)
        self.out_queue = Queue(-1)

        self.stop_event = Event()

        self.thread = Thread(target=self.run, args=[], daemon=True)
        self.thread.start()

    def run(self):
        # Create a MediaPipe pose instance
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        ) as pose:
            while not self.stop_event.set():
                frame = self.in_queue.get()

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
                point_packet = PointPacket(point_ids, landmark_xy)

                self.out_queue.put(point_packet)

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def get_points(self, frame: np.ndarray) -> PointPacket:
        """ """
        self.in_queue.put(frame)
        point_packet = self.out_queue.get()

        return point_packet

    def get_point_names(self, point_id) -> str:
        return POINT_NAMES[point_id]

    def draw_instructions(self, point_id: int) -> dict:
        if self.get_point_names(point_id).startswith("left"):
            rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        elif self.get_point_names(point_id).startswith("right"):
            rules = {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        else: 
            rules = {"radius": 5, "color": (220, 0, 220), "thickness": 3}

        return rules

