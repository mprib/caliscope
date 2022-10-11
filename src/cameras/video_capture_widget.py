# This widget is the primary functional unit of the motion capture. It
# establishes the connection with the video source 

import queue
from threading import Thread
import cv2
import time
import sys
import mediapipe as mp
import numpy as np

from datetime import datetime

# Append main repo to top of path to allow import of backend
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera

class CameraCaptureWidget:
    def __init__(self, cam):

        self.cam = cam 
        # Initialize parameters capture paramters
        self.rotation_count = 0 # +1 for each 90 degree clockwise rotation, -1 for CCW
        
        # Start the thread to read frames from the video stream
        self.cap_thread = Thread(target=self.roll_camera, args=( ), daemon=True)
        self.cap_thread.start()
        self.frame_name = "Cam"+str(cam.port)
        
        # initialize time trackers for actual FPS determination
        self.start_time = time.time()
        self.avg_delta_time = None

        # Mediapipe hand detection infrastructure
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils 
        self.show_mediapipe = True

    def get_FPS_actual(self):
        """set the actual frame rate from within the update function"""
        self.delta_time = time.time() - self.start_time
        self.start_time = time.time()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time

        # folding in current frame rate to trailing average to smooth out
        self.avg_delta_time = 0.8*self.avg_delta_time + 0.2*self.delta_time
        self.previous_time = self.start_time

        return 1/self.avg_delta_time

    def apply_rotation(self):

        if self.rotation_count == 0:
            pass
        elif self.rotation_count in [1, -3]:
            self._working_frame = cv2.rotate(self._working_frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_count in [2,-2]:
            self._working_frame = cv2.rotate(self._working_frame, cv2.ROTATE_180)
        elif self.rotation_count in [-1, 3]:
            self._working_frame = cv2.rotate(self._working_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def rotate_CW(self):
        print("Rotate CW")
        if self.rotation_count == 3:
            self.rotation_count = 0
        else:
            self.rotation_count = self.rotation_count + 1

    def rotate_CCW(self):
        print("Rotate CCW")
        if self.rotation_count == -3:
            self.rotation_count = 0
        else:
            self.rotation_count = self.rotation_count - 1

    def run_mediapipe_hands(self):

         # Only calculate mediapipe if going to display it
        if self.show_mediapipe:
            frame_RGB  = cv2.cvtColor(self._working_frame, cv2.COLOR_BGR2RGB)
            self.hand_results = self.hands.process(frame_RGB)
        
            # draw hand dots and lines
            if self.hand_results.multi_hand_landmarks:
                for handLms in self.hand_results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(self._working_frame, handLms, self.mpHands.HAND_CONNECTIONS)

    def roll_camera(self):
        """
        Worker function that is spun up by Thread. Reads in a working frame, 
        calls various frame processing methods on it, and updates the exposed 
        frame

        """
        # Grab frame and run image detection
        while True:
            self.cam.is_rolling = True

            if self.cam.capture.isOpened(): # note this line is truly necessary otherwise error upon closing capture
                # pull in a working frame
                (self.status, self._working_frame) = self.cam.capture.read()

                self.apply_rotation()
                self.run_mediapipe_hands()
                
                self.frame = self._working_frame.copy()
                self.FPS_actual = self.get_FPS_actual() 

                # Stop thread if camera pulls trigger
                if self.cam.stop_rolling_trigger:
                    self.cam.is_rolling = False
                    break

    def change_resolution(self, res):
        self.cam.stop_rolling() # will trigger running capture thread to end
        blank_image = np.zeros(self.frame.shape, dtype=np.uint8)
        self.frame = blank_image
        
        while self.cam.is_rolling:
            time.sleep(.01)

        self.cam.disconnect()

        self.cam.connect()
        self.cam.resolution = res
        
        # apparently threads can only be started once, so create anew
        self.cap_thread = Thread(target=self.roll_camera, args=( ), daemon=True)
        self.cap_thread.start()

    def toggle_mediapipe(self):
        self.show_mediapipe = not self.show_mediapipe
                
    
    def add_fps(self):
        """NOTE: this is used in main(), not in external use fo this module"""
        self.fps_text =  str(int(round(self.FPS_actual, 0))) 
        cv2.putText(self.frame, "FPS:" + self.fps_text, (10, 70),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 3)
        

# Highlight module functionality. View a frame with mediapipe hands
# press "q" to quit
if __name__ == '__main__':
    ports = [0]
    # ports = [0, 1, 3]

    cams = []
    for port in ports:
        print(f"Creating camera {port}")
        cams.append(Camera(port))

    camcap_widgets = []

    for cam in cams:
        print(f"Creating capture widget for camera {cam.port}")
        camcap_widgets.append(CameraCaptureWidget(cam))

    while True:
        try:
            for camcap in camcap_widgets:
                camcap.add_fps()
                cv2.imshow(camcap.frame_name, camcap.frame)
                
        # bad reads until connection to src established
        except AttributeError:
            pass

        key = cv2.waitKey(1)

        # toggle mediapipe with 'm' 
        if key == ord('m'):
            print("Toggling Mediapipe")
            for camcap in camcap_widgets:
                print(camcap.frame_name)
                camcap.toggle_mediapipe()
        
        if key == ord('r'):
            print("Rotate Frame CW")

            for camcap in camcap_widgets:
                camcap.rotate_CW()
                print(camcap.frame_name + " " + str(camcap.rotation_count))
       
        if  key == ord('l'):
            print("Rotate Frame CCW")
                
            for camcap in camcap_widgets:
                camcap.rotate_CCW()
                print(camcap.frame_name + " " + str(camcap.rotation_count))
       
        # 'q' to quit
        if key == ord('q'):
            for camcap in camcap_widgets:
                camcap.cam.capture.release()
            cv2.destroyAllWindows()
            exit(0)

        if key == ord('v'):
            for camcap in camcap_widgets:
                camcap.change_resolution((1280, 720))