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


class FaceTracker(Tracker):
    # Initialize MediaPipe Facemeshes and Drawing utility
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
        return "FACE"

    def run_frame_processor(self, port: int, rotation_count: int):
        # Create a MediaPipe FaceMesh instance
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        ) as facemeshes:
            while True:
                frame = self.in_queues[port].get()
                # apply rotation as needed
                frame = apply_rotation(frame, rotation_count)

                height, width, color = frame.shape
                # Convert the image to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = facemeshes.process(frame)

                # initialize variables so none will be created if no points detected
                point_ids = []
                landmark_xy = []

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for landmark_id, landmark in enumerate(face_landmarks.landmark):
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

    def get_point_name(self, point_id: int) -> str:
        return str(point_id)

    def scatter_draw_instructions(self, point_id: int) -> dict:
        if POINT_ID2NAME[point_id] == "silhouette":
            rules = {"radius": 5, "color": (0, 0, 0), "thickness": 3}
        if POINT_ID2NAME[point_id] in ("lipsUpperOuter", "lipsLowerOuter", "lipsUpperInner", "lipsLowerInner"):
            rules = {"radius": 5, "color": (128, 0, 0), "thickness": 3}
        if POINT_ID2NAME[point_id] in (
            "rightEyeUpper0",
            "rightEyeLower0",
            "rightEyeUpper1",
            "rightEyeLower1",
            "rightEyeUpper2",
            "rightEyeLower2",
            "rightEyeLower3",
            "rightEyeIris",
        ):
            rules = {"radius": 5, "color": (0, 255, 0), "thickness": 3}
        if POINT_ID2NAME[point_id] in ("rightEyebrowUpper", "rightEyebrowLower"):
            rules = {"radius": 5, "color": (0, 128, 0), "thickness": 3}
        if POINT_ID2NAME[point_id] in (
            "leftEyeUpper0",
            "leftEyeLower0",
            "leftEyeUpper1",
            "leftEyeLower1",
            "leftEyeUpper2",
            "leftEyeLower2",
            "leftEyeLower3",
            "leftEyeIris",
        ):
            rules = {"radius": 5, "color": (255, 0, 0), "thickness": 3}
        if POINT_ID2NAME[point_id] in ("leftEyebrowUpper", "leftEyebrowLower"):
            rules = {"radius": 5, "color": (128, 0, 0), "thickness": 3}
        if POINT_ID2NAME[point_id] == "midwayBetweenEyes":
            rules = {"radius": 5, "color": (0, 0, 0), "thickness": 3}
        if POINT_ID2NAME[point_id] == (
            "noseTip",
            "noseBottom",
            "noseRightCorner",
            "noseLeftCorner",
            "leftCheek",
            "rightCheek",
        ):
            rules = {"radius": 5, "color": (0, 0, 255), "thickness": 3}
        else:
            rules = {"radius": 5, "color": (0, 0, 0), "thickness": 3}
        return rules


# Keypoint names copied from tfjs under Apache License 2.0, "AS-IS".
NAME2KEYPOINTS = {
    "silhouette": [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
    "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
    "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
    "rightEyeIris": [473, 474, 475, 476, 477],
    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
    "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
    "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
    "leftEyeIris": [468, 469, 470, 471, 472],
    "midwayBetweenEyes": [168],
    "noseTip": [1],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
    "rightCheek": [205],
    "leftCheek": [425],
}

NAMES = sorted(NAME2KEYPOINTS)
POINT_ID2NAME = [None for _ in range(478)]
for i_n, name in enumerate(NAMES):
    for id in NAME2KEYPOINTS[name]:
        POINT_ID2NAME[id] = name
