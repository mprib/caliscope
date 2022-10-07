import queue
from threading import Thread
import cv2
import time
import sys
import mediapipe as mp

from datetime import datetime


class VideoCaptureWidget:
    def __init__(self, src):
        
        # Initialize parameters capture paramters
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # from https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-getanother thread signaled a change to mediapipe overley-sync
        self.rotation_count = 0 # +1 for each 90 degree clockwise rotation, -1 for CCW
        
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Create queues used to pass parameters into the worker thread
        self.mp_toggle_q = queue.Queue()
        self.rotation_q = queue.Queue()

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=(self.mp_toggle_q, ))
        self.thread.daemon = True
        self.thread.start()
        self.frame_name = "Cam"+str(src)
        
        # initialize time trackers for actual FPS determination
        self.start_time = time.time()
        self.avg_delta_time = None

        # Mediapipe hand detection infrastructure
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils 
        self.show_medipipe = True

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

    def apply_rotation(self, raw_frame):

        if self.rotation_count == 0:
            return raw_frame
        elif self.rotation_count in [1, -3]:
            return cv2.rotate(raw_frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_count in [2,-2]:
            return cv2.rotate(raw_frame, cv2.ROTATE_180)
        elif self.rotation_count in [-1, 3]:
            return cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

    def update(self, show_mp_q):
        """
        Worker function that is spun up by Thread. This seems to be where much
        of the substantive processing and real-time configuration will occur

        I'm not sure if this would map directly to post-processing tasks, but it
        will probably be pretty close.

        Parameters:
            - mp_toggle_q: a queue passed to the thread that will signal a 
            change of self.show_mediapipe
        """
        # Grap frame and run image detection
        while True:
            
            # check to see if mediapipe should be generated
            if not show_mp_q.empty():
                self.show_medipipe = show_mp_q.get()

            if self.capture.isOpened(): # note this line is truly necessary otherwise error upon closing capture
                # pull in a working frame
                (self.status, working_frame) = self.capture.read()

                working_frame = self.apply_rotation(working_frame)

                # Only calculate mediapipe if going to display it
                if self.show_medipipe:
                    frame_RGB  = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
                    self.hand_results = self.hands.process(frame_RGB)

                self.frame = working_frame.copy() 
                
                # only draw hand landmarks if calculating mediapipe
                if self.show_medipipe:
                    # draw hand dots and lines
                    if self.hand_results.multi_hand_landmarks:
                        for handLms in self.hand_results.multi_hand_landmarks:
                            self.mpDraw.draw_landmarks(self.frame, handLms, self.mpHands.HAND_CONNECTIONS)

                # wait to read next frame in order to hit target FPS. Record FPS
                self.FPS_actual = self.get_FPS_actual() 
    
    def toggle_mediapipe(self):

        if self.show_medipipe == True:
            self.mp_toggle_q.put(False)
        else:
            self.mp_toggle_q.put(True)
                
    
    def grab_frame(self):
        
        self.fps_text =  str(int(round(self.FPS_actual, 0))) 
        self.time_now = str(datetime.now().strftime("%S"))
        self.sec_now = self.time_now[1]

        cv2.putText(self.frame, "FPS:" + self.fps_text, (10, 70),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 3)
        

# Highlight module functionality. View a frame with mediapipe hands
# press "q" to quit
if __name__ == '__main__':
    src_list = [0,1]
    # src_list = [0]
    cam_widgets = []

    for src in src_list:
        cam_widgets.append(VideoCaptureWidget(src))

    while True:
        try:
            for cam in cam_widgets:
                cam.grab_frame()
                cv2.imshow(cam.frame_name, cam.frame)
                
        except AttributeError:
            pass

        key = cv2.waitKey(1)

        # toggle mediapipe with 'm' 
        if key == ord('m'):
            print("Toggling Mediapipe")
            for cam in cam_widgets:
                print(cam.frame_name)
                cam.toggle_mediapipe()
        
        if key == ord('r'):
            print("Rotate Frame CW")

            for cam in cam_widgets:
                cam.rotate_CW()
                print(cam.frame_name + " " + str(cam.rotation_count))
       
        if  key == ord('l'):
            print("Rotate Frame CCW")
                
            for cam in cam_widgets:
                cam.rotate_CCW()
                print(cam.frame_name + " " + str(cam.rotation_count))
       
        # 'q' to quit
        if key == ord('q'):
            for cam in cam_widgets:
                cam.capture.release()
            cv2.destroyAllWindows()
            exit(0)
