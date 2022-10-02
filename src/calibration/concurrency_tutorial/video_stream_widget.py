from threading import Thread
import cv2, time
import sys
import detect_2D_points


class VideoStreamWidget(object):
    def __init__(self, src, width, height):
        self.FPS_target = 40

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
        self.current_time = 0
        self.previous_time = 0

    def get_FPS_actual(self):
        """set the actual frame rate from within the update function"""
        self.current_time = time.time()
        delta_time = self.current_time-self.previous_time
        self.previous_time = self.current_time

        return 1/delta_time



    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            # wait to read next frame in order to hit target FPS. Record FPS
            detect_2D_points.get_hands(self.frame)
            self.FPS_actual = self.get_FPS_actual() 
            time.sleep(1/self.FPS_target)
    
    def show_frame(self):
        # Display frames in main program
        cv2.putText(self.frame, str(int(round(self.FPS_actual, 0))), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255), 3)
        cv2.imshow(self.frame_name, self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(0)

if __name__ == '__main__':
# cam 0: Device USB\VID_328F&PID_003F&MI_00\6&124b7a38&0&0000 was started.
# cam 1: Device USB\VID_328F&PID_003F&MI_00\6&b0660f1&0&0000 was started.
# Don't get bogged down in this at the moment, Mac. Pulling unique camera
# identifiers is going to be an operating system dependent thing, and so 
# you shouldn't get too bogged down in it right now.

    src_list = [0,1]
    # src_list = [0]
    cam_widgets = []
    for src in src_list:
        cam_widgets.append(VideoStreamWidget(src, 1080, 640))

    while True:
        try:
            for cam in cam_widgets:
                cam.show_frame()
        except AttributeError:
            pass