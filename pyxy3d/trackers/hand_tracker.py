"""
This is a bit of an initial volley at point tracking just to have the 
basics of something to throw at pyxy3d as a basic test of integrating a
streamlined point tracking manager that could be expanded out further.

Wondering now about the name LabSeurat. Or heck, just stick with Seurat.

"""

import mediapipe as mp
import numpy as np
import cv2
# cap = cv2.VideoCapture(0)
from pyxy3d.interface import Tracker, PointPacket



class HandTracker(Tracker):
    # Initialize MediaPipe Hands and Drawing utility
    def __init__(self) -> None:
        # mp.solutions.hands = mp.solutions.hands
        # mp_drawing = mp.solutions.drawing_utils

        # Create a MediaPipe Hands instance
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_points(self, frame:np.ndarray)->PointPacket:
        height, width, color = frame.shape
        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(image_rgb)
        # initialize variables so none will be created if no points detected


        point_ids = []
        landmark_xy = []
        
        if results.multi_hand_landmarks:
            # need to track left/right which is a but more 
            # difficult than you might think
            hand_types = [] 
            for item in results.multi_handedness:
                hand_info = item.ListFields()[0][1].pop()
                hand_types.append(hand_info.label)

            hand_type_index = 0

            for hand_landmarks in results.multi_hand_landmarks:
                
                # create adjusting factor to distinguish 
                # left/right 
                hand_label = hand_types[hand_type_index]
                if hand_label == "Left":
                    side_adjustment_factor = 0
                else:
                    side_adjustment_factor = 100 

                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    point_ids.append(landmark_id+side_adjustment_factor)

                    # mediapipe expresses in terms of percent of frame, so must map to pixel position
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    landmark_xy.append((x, y))

                hand_type_index+=1
                
        point_ids = np.array(point_ids)
        landmark_xy = np.array(landmark_xy)

        return PointPacket(point_ids,landmark_xy)
            

    def get_point_names(self) -> dict:
        return super().get_point_names()
    
    def draw_instructions(self, point_id:int)->dict:
        if point_id < 100:
            rules = {"radius":5,
                     "color":(0,0,220),
                     "thickness":3}
        else:
            rules = {"radius":5,
                     "color":(220,0,0),
                     "thickness":3}
        return rules
