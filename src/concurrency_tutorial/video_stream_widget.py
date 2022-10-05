from re import I
from threading import Thread
import cv2, time
import sys
import mediapipe as mp

from datetime import datetime


# import detect_2D_points

class VideoCaptureWidget:
    def __init__(self, src, width, height):
        self.FPS_target = 50

        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # from https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)

        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.frame_name = "Cam"+str(src)

        # initialize time trackers of frame updates
        self.start_time = time.time()
        # self.previous_time = 0
        self.avg_delta_time = None

        # Mediapipe hand detection infrastructure
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils 
        self.frame_processed = False
        # self.connected_landmarks = self.mpHands.HAND_CONNECTIONS

    def get_FPS_actual(self):
        """set the actual frame rate from within the update function"""
        self.delta_time = time.time() - self.start_time
        self.start_time = time.time()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time
        self.avg_delta_time = 0.8*self.avg_delta_time + 0.2*self.delta_time
        self.previous_time = self.start_time

        return 1/self.avg_delta_time
    

    def update(self):
        """
        Worker function that is spun up by Thread
        """
        # Grap frame and run image detection
        while True:
            if self.capture.isOpened():

                # pull in working frame
                (self.status, working_frame) = self.capture.read()
                frame_RGB  = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
                self.hand_results = self.hands.process(frame_RGB)
                            
                # draw hand dots and lines
                self.frame_processed = False
                self.frame = working_frame.copy() 
                if self.hand_results.multi_hand_landmarks:
                    for handLms in self.hand_results.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(self.frame, handLms, self.mpHands.HAND_CONNECTIONS)
                self.frame_processed = True 
                # wait to read next frame in order to hit target FPS. Record FPS
                self.FPS_actual = self.get_FPS_actual() 
                time.sleep(1/self.FPS_target)    #TODO: adjust sleep time to incorporate worker loop time
    


    def grab_frame(self):
        
        # display_text = "FPS:" + fps_text + "Time:" + time_now
        fps_text =  str(int(round(self.FPS_actual, 0))) 
        time_now = str(datetime.now().strftime("%H:%M:%S"))
        cv2.putText(self.frame, "FPS:" + fps_text, (10, 70),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 3)
        cv2.putText(self.frame, "Time:" + time_now, (10,140),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 3)
        
# Highlight module functionality. View a frame with mediapipe hands
# press "q" to quit
if __name__ == '__main__':
    # src_list = [0,1]
    src_list = [0]
    cam_widgets = []

    for src in src_list:
        cam_widgets.append(VideoCaptureWidget(src, 1080, 640))

    while True:
        try:
            for cam in cam_widgets:
                cam.grab_frame()
                cv2.imshow(cam.frame_name, cam.frame)
                
        except AttributeError:
            pass
     
        if cv2.waitKey(1) == ord('q'):
            for cam in cam_widgets:
                cam.capture.release()
            cv2.destroyAllWindows()
            exit(0)