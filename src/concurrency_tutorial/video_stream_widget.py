import queue
from re import I
from threading import Thread
import cv2, time
import sys
import mediapipe as mp


from datetime import datetime


# import detect_2D_points

class VideoCaptureWidget:
    def __init__(self, src, width, height, mp_toggle_q ):
        self.FPS_target = 50
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # from https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.mp_toggle_q = mp_toggle_q

        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=(mp_toggle_q, ))
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
        self.show_medipipe = True

    def get_FPS_actual(self):
        """set the actual frame rate from within the update function"""
        self.delta_time = time.time() - self.start_time
        self.start_time = time.time()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time
        self.avg_delta_time = 0.8*self.avg_delta_time + 0.2*self.delta_time
        self.previous_time = self.start_time

        return 1/self.avg_delta_time
    

    def update(self, mp_toggle_q):
        """
        Worker function that is spun up by Thread

        Parameters:
            - mp_toggle_q: a queue passed to the thread that will signal a 
            change of self.show_mediapipe
        """
        # Grap frame and run image detection
        while True:
            
            # check to see if another thread signaled a change to mediapipe overley
            if not mp_toggle_q.empty():
                self.show_medipipe = mp_toggle_q.get()

            if self.capture.isOpened():
                # pull in working frame
                (self.status, working_frame) = self.capture.read()

                frame_RGB  = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)

                # Only calculate mediapipe if going to display it
                if self.show_medipipe:
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

        print(self.show_medipipe)
        if self.show_medipipe == True:
            self.show_medipipe == False
        else:
            self.show_medipipe == True
                
        print(self.show_medipipe)
    
    def grab_frame(self):
        
        self.fps_text =  str(int(round(self.FPS_actual, 0))) 
        self.time_now = str(datetime.now().strftime("%S"))
        self.sec_now = self.time_now[1]

        # if int(self.sec_now) < 5:
        #     self.show_medipipe = False
        # else:
        #     self.show_medipipe = True

        cv2.putText(self.frame, "FPS:" + self.fps_text, (10, 70),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 3)
        cv2.putText(self.frame, "Time:" + self.sec_now, (10,140),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 3)
        

# Highlight module functionality. View a frame with mediapipe hands
# press "q" to quit
if __name__ == '__main__':
    src_list = [0,1]
    # src_list = [0]
    cam_widgets = []
    q_s = []
    for src in src_list:
        q = queue.Queue()
        q_s.append(q)   
        cam_widgets.append(VideoCaptureWidget(src, 640, 480, q))

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
        if cv2.waitKey(1) == ord('m'):
            for cam in cam_widgets:

                cam.capture.release()
            cv2.destroyAllWindows()
            exit(0)