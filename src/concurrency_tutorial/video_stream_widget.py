from re import I
from threading import Thread
import cv2, time
import sys
import mediapipe as mp


# import detect_2D_points

class VideoCaptureWidget:
    def __init__(self, src, width, height):
        self.FPS_target = 30

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
        self.connected_landmarks = self.mpHands.HAND_CONNECTIONS

    def get_FPS_actual(self):
        """set the actual frame rate from within the update function"""
        self.delta_time = time.time() - self.start_time
        self.start_time = time.time()
        if not self.avg_delta_time:
            self.avg_delta_time = self.delta_time
        self.avg_delta_time = 0.8*self.avg_delta_time + 0.2*self.delta_time
        self.previous_time = self.start_time

        return 1/self.avg_delta_time
    
    def find_landmarks(self):
        imgRGB  = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)
        self.hand_results = self.hands.process(imgRGB)


    def update(self):
        # Grap frame and run image detection
        while True:
            if self.capture.isOpened():
                (self.status, self.raw_frame) = self.capture.read()
                # wait to read next frame in order to hit target FPS. Record FPS
                self.find_landmarks()
                self.FPS_actual = self.get_FPS_actual() 
                time.sleep(1/self.FPS_target)
    
    def grab_frame(self):
        # draw hand dots and lines
        if self.hand_results.multi_hand_landmarks:
            for handLms in self.hand_results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(self.raw_frame, handLms, self.mpHands.HAND_CONNECTIONS)

        # Display frames in main program
        display_text = "FPS:" + str(int(round(self.FPS_actual, 0)))
        cv2.putText(self.raw_frame, display_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,0,0), 2)
        

def main():

    src_list = [0,1]
    # src_list = [0]
    cam_widgets = []
    for src in src_list:
        cam_widgets.append(VideoCaptureWidget(src, 1080, 640))

    while True:
        try:
            for cam in cam_widgets:
                cam.grab_frame()
                cv2.imshow(cam.frame_name, cam.raw_frame)
                
        except AttributeError:
            pass
     
        if cv2.waitKey(1) == ord('q'):
            for cam in cam_widgets:
                cam.capture.release()
            cv2.destroyAllWindows()
            exit(0)

if __name__ == '__main__':
    main()