import caliscope.logger

from threading import Thread
from queue import Queue

import mediapipe as mp
import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points
logger = caliscope.logger.get(__name__)

class HandTracker(Tracker):
    # Initialize MediaPipe Hands and Drawing utility
    def __init__(self) -> None:
        self.in_queue = Queue(-1)
        self.out_queue = Queue(-1)

        # each port gets its own mediapipe context manager
        # use a dictionary of queues for passing 
        self.in_queues = {}
        self.out_queues = {}
        self.threads = {}


    @property
    def name(self):
        return "HAND"

    def run_frame_processor(self, port: int, rotation_count: int):
        # Create a MediaPipe Hands instance
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        ) as hands:
            while True:
                frame = self.in_queues[port].get()
                # apply rotation as needed
                frame = apply_rotation(frame, rotation_count)

                height, width, color = frame.shape
                # Convert the image to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame)

                # initialize variables so none will be created if no points detected
                point_ids = []
                landmark_xy = []

                if results.multi_hand_landmarks:
                    # need to track left/right...more difficult than you might think
                    hand_types = []
                    for item in results.multi_handedness:
                        hand_info = item.ListFields()[0][1].pop()
                        hand_types.append(hand_info.label)

                    hand_type_index = 0

                    for hand_landmarks in results.multi_hand_landmarks:
                        # create adjusting factor to distinguish left/right
                        hand_label = hand_types[hand_type_index]
                        if hand_label == "Left":
                            side_adjustment_factor = 0
                        else:
                            side_adjustment_factor = 100

                        for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                            point_ids.append(landmark_id + side_adjustment_factor)

                            # mediapipe expresses in terms of percent of frame, so must map to pixel position
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            landmark_xy.append((x, y))

                        hand_type_index += 1

                point_ids = np.array(point_ids)
                landmark_xy = np.array(landmark_xy)
                landmark_xy = unrotate_points(landmark_xy, rotation_count, width,height)

                point_packet = PointPacket(point_ids, landmark_xy)

                self.out_queues[port].put(point_packet)


    def get_points(
        self, frame: np.ndarray, port: int, rotation_count: int
    ) -> PointPacket:
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
        return str(point_id)

    def scatter_draw_instructions(self, point_id: int) -> dict:
        if point_id < 100:
            rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        else:
            rules = {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        return rules

