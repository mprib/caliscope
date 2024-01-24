import pyxy3d.logger

from threading import Thread
from queue import Queue

import mediapipe as mp
import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
from pyxy3d.packets import PointPacket
from pyxy3d.tracker import Tracker, WireFrameView, Segment
from pyxy3d.trackers.helper import apply_rotation, unrotate_points

logger = pyxy3d.logger.get(__name__)

DRAW_IGNORE_LIST = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_wrist_pose",
    "right_wrist_pose",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
]

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
    15: "left_wrist_pose",
    16: "right_wrist_pose",
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
    100: "right_wrist",
    101: "right_thumb_CMC",
    102: "right_thumb_MCP",
    103: "right_thumb_IP",
    104: "right_thumb_tip",
    105: "right_index_finger_MCP",
    106: "right_index_finger_PIP",
    107: "right_index_finger_DIP",
    108: "right_index_finger_tip",
    109: "right_middle_finger_MCP",
    110: "right_middle_finger_PIP",
    111: "right_middle_finger_DIP",
    112: "right_middle_finger_tip",
    113: "right_ring_finger_MCP",
    114: "right_ring_finger_PIP",
    115: "right_ring_finger_DIP",
    116: "right_ring_finger_tip",
    117: "right_pinky_MCP",
    118: "right_pinky_PIP",
    119: "right_pinky_DIP",
    120: "right_pinky_tip",
    200: "left_wrist",
    201: "left_thumb_CMC",
    202: "left_thumb_MCP",
    203: "left_thumb_IP",
    204: "left_thumb_tip",
    205: "left_index_finger_MCP",
    206: "left_index_finger_PIP",
    207: "left_index_finger_DIP",
    208: "left_index_finger_tip",
    209: "left_middle_finger_MCP",
    210: "left_middle_finger_PIP",
    211: "left_middle_finger_DIP",
    212: "left_middle_finger_tip",
    213: "left_ring_finger_MCP",
    214: "left_ring_finger_PIP",
    215: "left_ring_finger_DIP",
    216: "left_ring_finger_tip",
    217: "left_pinky_MCP",
    218: "left_pinky_PIP",
    219: "left_pinky_DIP",
    220: "left_pinky_tip",
}


# keep ids in distinct ranges to avoid clashes
POSE_OFFSET = 0
RIGHT_HAND_OFFSET = 100
LEFT_HAND_OFFSET = 200
FACE_OFFSET = 500

### Skeleton Draw Instructions
pelvis = Segment(name="pelvis", color="y", point_A="right_hip", point_B="left_hip")
right_flank = Segment(
    name="right_flank", color="y", point_A="right_hip", point_B="right_shoulder"
)
left_flank = Segment(
    name="left_flank", color="y", point_A="left_hip", point_B="left_shoulder"
)
shoulders = Segment(
    name="shoulder", color="y", point_A="right_shoulder", point_B="left_shoulder"
)

right_thigh = Segment(
    name="right_thigh", color="r", point_A="right_hip", point_B="right_knee"
)
left_thigh = Segment(
    name="left_thigh", color="g", point_A="left_hip", point_B="left_knee"
)
right_shank = Segment(
    name="right_shank", color="r", point_A="right_knee", point_B="right_ankle"
)
left_shank = Segment(
    name="left_shank", color="g", point_A="left_knee", point_B="left_ankle"
)
right_arm = Segment(
    name="right_arm", color="r", point_A="right_shoulder", point_B="right_elbow"
)
right_forearm = Segment(
    name="right_forearm", color="r", point_A="right_elbow", point_B="right_wrist"
)
left_arm = Segment(
    name="left_arm", color="g", point_A="left_shoulder", point_B="left_elbow"
)
left_forearm = Segment(
    name="left_forearm", color="g", point_A="left_elbow", point_B="left_wrist"
)

# Right Hand Segments
right_thumb = Segment(
    name="right_thumb", color="r", point_A="right_wrist", point_B="right_thumb_CMC"
)
right_thumb_proximal = Segment(
    name="right_thumb_proximal", color="r", point_A="right_thumb_CMC", point_B="right_thumb_MCP"
)
right_thumb_intermediate = Segment(
    name="right_thumb_intermediate", color="r", point_A="right_thumb_MCP", point_B="right_thumb_IP"
)
right_thumb_distal = Segment(
    name="right_thumb_distal", color="r", point_A="right_thumb_IP", point_B="right_thumb_tip"
)

# ... Similar segments for other fingers of the right hand ...

# Left Hand Segments
left_thumb = Segment(
    name="left_thumb", color="g", point_A="left_wrist", point_B="left_thumb_CMC"
)
left_thumb_proximal = Segment(
    name="left_thumb_proximal", color="g", point_A="left_thumb_CMC", point_B="left_thumb_MCP"
)
left_thumb_intermediate = Segment(
    name="left_thumb_intermediate", color="g", point_A="left_thumb_MCP", point_B="left_thumb_IP"
)
left_thumb_distal = Segment(
    name="left_thumb_distal", color="g", point_A="left_thumb_IP", point_B="left_thumb_tip"
)

# ... Similar segments for other fingers of the left hand ...

# Feet Segments
right_foot = Segment(
    name="right_foot", color="r", point_A="right_ankle", point_B="right_heel"
)
right_toe = Segment(
    name="right_toe", color="r", point_A="right_heel", point_B="right_foot_index"
)

left_foot = Segment(
    name="left_foot", color="g", point_A="left_ankle", point_B="left_heel"
)
left_toe = Segment(
    name="left_toe", color="g", point_A="left_heel", point_B="left_foot_index"
)


HOLISTIC_SKELETON = WireFrameView(
    segments=[
        pelvis,
        right_flank,
        left_flank,
        shoulders,
        right_thigh,
        left_thigh,
        right_shank,
        left_shank,
        right_arm,
        left_arm,
        right_forearm,
        left_forearm,
    right_thumb, right_thumb_proximal, right_thumb_intermediate, right_thumb_distal,
    left_thumb, left_thumb_proximal, left_thumb_intermediate, left_thumb_distal,
    right_foot, right_toe, left_foot, left_toe
    ],
    point_names=POINT_NAMES,
)


###
class HolisticTracker(Tracker):
    def __init__(self) -> None:
        # each port gets its own mediapipe context manager
        # use a dictionary of queues for passing
        self.in_queues = {}
        self.out_queues = {}
        self.threads = {}
        self.wireframe = HOLISTIC_SKELETON

    @property
    def name(self):
        return "HOLISTIC"

    def run_frame_processor(self, port: int, rotation_count: int):
        # Create a MediaPipe pose instance
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.8, min_tracking_confidence=0.8
        ) as holistic:
            while True:
                frame = self.in_queues[port].get()
                # apply rotation as needed
                frame = apply_rotation(frame, rotation_count)

                height, width, color = frame.shape
                # Convert the image to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame)

                # initialize variables so none will be created if no points detected
                point_ids = []
                landmark_xy = []

                if results.pose_landmarks:
                    for landmark_id, landmark in enumerate(
                        results.pose_landmarks.landmark
                    ):
                        # mediapipe expresses in terms of percent of frame, so must map to pixel position
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        if (
                            landmark.x < 0
                            or landmark.x > 1
                            or landmark.y < 0
                            or landmark.y > 1
                        ):
                            # ignore
                            pass
                        else:
                            point_ids.append(landmark_id + POSE_OFFSET)
                            landmark_xy.append((x, y))

                if results.right_hand_landmarks:
                    for landmark_id, landmark in enumerate(
                        results.right_hand_landmarks.landmark
                    ):
                        # mediapipe expresses in terms of percent of frame, so must map to pixel position
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        if (
                            landmark.x < 0
                            or landmark.x > 1
                            or landmark.y < 0
                            or landmark.y > 1
                        ):
                            # ignore
                            pass
                        else:
                            point_ids.append(landmark_id + RIGHT_HAND_OFFSET)
                            landmark_xy.append((x, y))

                if results.left_hand_landmarks:
                    for landmark_id, landmark in enumerate(
                        results.left_hand_landmarks.landmark
                    ):
                        # mediapipe expresses in terms of percent of frame, so must map to pixel positionND_OFFSET
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        if (
                            landmark.x < 0
                            or landmark.x > 1
                            or landmark.y < 0
                            or landmark.y > 1
                        ):
                            # ignore
                            pass
                        else:
                            point_ids.append(landmark_id + LEFT_HAND_OFFSET)
                            landmark_xy.append((x, y))

                if results.face_landmarks:
                    for landmark_id, landmark in enumerate(
                        results.face_landmarks.landmark
                    ):
                        # mediapipe expresses in terms of percent of frame, so must map to pixel positionFSET
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        if (
                            landmark.x < 0
                            or landmark.x > 1
                            or landmark.y < 0
                            or landmark.y > 1
                        ):
                            # ignore
                            pass
                        else:
                            point_ids.append(landmark_id + FACE_OFFSET)
                            landmark_xy.append((x, y))

                point_ids = np.array(point_ids)
                landmark_xy = np.array(landmark_xy)
                landmark_xy = unrotate_points(
                    landmark_xy, rotation_count, width, height
                )
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

    def get_point_name(self, point_id) -> str:
        if point_id < FACE_OFFSET:
            point_name = POINT_NAMES[point_id]
        else:
            point_name = "face_" + str(point_id - FACE_OFFSET)
        return point_name

    def scatter_draw_instructions(self, point_id: int) -> dict:
        point_name = self.get_point_name(point_id)
        if point_name in DRAW_IGNORE_LIST:
            rules = {"radius": 0, "color": (0, 0, 0), "thickness": 0}
        elif point_name.startswith("left"):
            rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        elif point_name.startswith("right"):
            rules = {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        else:
            rules = {"radius": 1, "color": (220, 0, 220), "thickness": 1}

        return rules

    def get_connected_points(self) -> set[tuple[int, int]]:
        return super().get_connected_points()
