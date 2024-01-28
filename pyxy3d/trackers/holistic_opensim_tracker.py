from threading import Thread
from queue import Queue

import mediapipe as mp
import numpy as np
import cv2

from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points

import caliscope.logger
logger = caliscope.logger.get(__name__)

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.95

# The following are from base Pose and can be ignored in favor of
# better estimated Holistic points
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
    # 0: "nose",
    # 1: "left_eye_inner",
    # 2: "left_eye",
    # 3: "left_eye_outer",
    # 4: "right_eye_inner",
    # 5: "right_eye",
    # 6: "right_eye_outer",
    # 7: "left_ear",
    # 8: "right_ear",
    # 9: "mouth_left",
    # 10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    # 15: "left_wrist_pose",
    # 16: "right_wrist_pose",
    # 17: "left_pinky",
    # 18: "right_pinky",
    # 19: "left_index",
    # 20: "right_index",
    # 21: "left_thumb",
    # 22: "right_thumb",
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
    # HOLISTIC FACE keypoints worth keeping...
    # this is for kinematic skull tracking, not animation.
    500: "lip_top_mid",
    504: "nose_tip",
    633: "right_inner_eye",
    699: "chin_tip",
    746: "right_outer_eye",
    862: "left_inner_eye",
    966: "left_outer_eye",
}


METARIG_BILATERAL_MEAUSURES = {
    "Hip_Shoulder_Distance": ["hip", "shoulder"],
    "Shoulder_Inner_Eye_Distance": ["inner_eye", "shoulder"],
    "Palm": ["index_finger_MCP", "pinky_MCP"],
    "Foot": ["heel", "foot_index"],
    "Upper_Arm": ["shoulder", "elbow"],
    "Forearm": ["elbow", "wrist"],
    "Wrist_to_MCP1": ["wrist", "thumb_MCP"],
    "Wrist_to_MCP2": ["wrist", "index_finger_MCP"],
    "Wrist_to_MCP3": ["wrist", "middle_finger_MCP"],
    "Wrist_to_MCP4": ["wrist", "ring_finger_MCP"],
    "Wrist_to_MCP5": ["wrist", "pinky_MCP"],
    "Prox_Phalanx_1": ["thumb_MCP", "thumb_IP"],
    "Prox_Phalanx_2": ["index_finger_MCP", "index_finger_PIP"],
    "Prox_Phalanx_3": ["middle_finger_MCP", "middle_finger_PIP"],
    "Prox_Phalanx_4": ["ring_finger_MCP", "ring_finger_PIP"],
    "Prox_Phalanx_5": ["pinky_MCP", "pinky_PIP"],
    "Mid_Phalanx_2": ["index_finger_PIP", "index_finger_DIP"],
    "Mid_Phalanx_3": ["middle_finger_PIP", "middle_finger_DIP"],
    "Mid_Phalanx_4": ["ring_finger_PIP", "ring_finger_DIP"],
    "Mid_Phalanx_5": ["pinky_PIP", "pinky_DIP"],
    "Dist_Phalanx_1": ["thumb_IP", "thumb_tip"],
    "Dist_Phalanx_2": ["index_finger_DIP", "index_finger_tip"],
    "Dist_Phalanx_3": ["middle_finger_DIP", "middle_finger_tip"],
    "Dist_Phalanx_4": ["ring_finger_DIP", "middle_finger_tip"],
    "Dist_Phalanx_5": ["pinky_DIP", "pinky_tip"],
    "Thigh_Length": ["hip", "knee"],
    "Shin_Length": ["knee", "ankle"],
}


METARIG_SYMMETRICAL_MEASURES = {
    "Shoulder_Width": ["left_shoulder", "right_shoulder"],
    "Hip_Width": ["left_hip", "right_hip"],
    "Inner_Eye_Distance": ["left_inner_eye", "right_inner_eye"],
}


# keep ids in distinct ranges to avoid clashes
POSE_OFFSET = 0
RIGHT_HAND_OFFSET = 100
LEFT_HAND_OFFSET = 200
FACE_OFFSET = 500


class HolisticOpenSimTracker(Tracker):
    def __init__(self) -> None:
        # each port gets its own mediapipe context manager
        # use a dictionary of queues for passing
        self.in_queues = {}
        self.out_queues = {}
        self.threads = {}

    @property
    def name(self):
        return "HOLISTIC_OPENSIM"

    @property
    def metarig_mapped(self):
        return True

    @property
    def metarig_symmetrical_measures(self):
        return METARIG_SYMMETRICAL_MEASURES

    @property
    def metarig_bilateral_measures(self):
        return METARIG_BILATERAL_MEAUSURES

    def run_frame_processor(self, port: int, rotation_count: int):
        # Create a MediaPipe pose instance
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
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
                            mapped_point_id = landmark_id + POSE_OFFSET
                            # some of the pose values are too noisy to bother with including considering that holistic face and hand tracking is so good
                            # ignore those points that aren't in the POINT_NAMES list
                            if mapped_point_id in POINT_NAMES:
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
                            face_id = landmark_id + FACE_OFFSET
                            # only track the point if it is in the list of names above
                            # this will significantly reduce the data tracked.
                            if face_id in POINT_NAMES.keys():
                                point_ids.append(landmark_id + FACE_OFFSET)
                                landmark_xy.append((x, y))

                point_ids = np.array(point_ids)
                landmark_xy = np.array(landmark_xy)

                # adjust for previous shift due to camera rotation count
                landmark_xy = unrotate_points(
                    landmark_xy, rotation_count, width, height
                )
                point_packet = PointPacket(point_ids, landmark_xy)

                self.out_queues[port].put(point_packet)

    def get_points(
        self, frame: np.ndarray, port: int, rotation_count: int
    ) -> PointPacket:
        """
        This is the primary method exposed to the rest of the code.
        The tracker receives frames and basic camera data from the Stream,
        then it places the frame/camera data on a queue that will hand it
        off to a context manager set up to process that stream of data.
        """

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
        # this if/else should be unnecessary now that only select points are being passed on up the chain.
        # if point_id < FACE_OFFSET:
        #     point_name = POINT_NAMES[point_id]
        # else:
        #     point_name = "face_" + str(point_id - FACE_OFFSET)
        return POINT_NAMES[point_id]

    def scatter_draw_instructions(self, point_id: int) -> dict:
        point_name = self.get_point_name(point_id)
        if point_name in DRAW_IGNORE_LIST:
            rules = {"radius": 0, "color": (0, 0, 0), "thickness": 0}
        elif point_name.startswith("left"):
            rules = {"radius": 5, "color": (0, 0, 220), "thickness": 3}
        elif point_name.startswith("right"):
            rules = {"radius": 5, "color": (220, 0, 0), "thickness": 3}
        else:
            rules = {"radius": 3, "color": (0, 220, 220), "thickness": 3}

        return rules
