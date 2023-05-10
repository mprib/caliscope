"""
This is a bit of an initial volley at point tracking just to have the 
basics of something to throw at pyxy3d as a basic test of integrating a
streamlined point tracking manager that could be expanded out further.

Wondering now about the name LabSeurat. Or heck, just stick with Seurat.

"""
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)
from threading import Thread, Event
from queue import Queue

import mediapipe as mp
import numpy as np
import cv2
# cap = cv2.VideoCapture(0)
from pyxy3d.interface import Tracker, PointPacket



class HandTracker(Tracker):
    # Initialize MediaPipe Hands and Drawing utility
    def __init__(self) -> None:
        
        self.in_queue = Queue(-1)
        self.out_queue = Queue(-1)


        self.stop_event = Event()
        
        self.thread = Thread(target=self.run, args=[],daemon=True)
        self.thread.start()
        
    def run(self):
        # Create a MediaPipe Hands instance
        with  mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            
            while not self.stop_event.set():
                frame = self.in_queue.get()

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
                            point_ids.append(landmark_id+side_adjustment_factor)

                            # mediapipe expresses in terms of percent of frame, so must map to pixel position
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            landmark_xy.append((x, y))

                        hand_type_index+=1
                
                point_ids = np.array(point_ids)
                landmark_xy = np.array(landmark_xy)
                point_packet = PointPacket(point_ids,landmark_xy)

                self.out_queue.put(point_packet)
                
                
                
    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def get_points(self, frame:np.ndarray)->PointPacket:

        self.in_queue.put(frame)
        point_packet = self.out_queue.get()
        
        return point_packet 

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
