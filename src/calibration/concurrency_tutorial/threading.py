from threading import Thread
import cv2, time
 
class VideoStreamWidget(object):
    def __init__(self, src):
        self.FPS = 30

        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # from https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.frame_name = "Cam"+str(src)

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(1/self.FPS)
    
    def show_frame(self):
        # Display frames in main program
        cv2.imshow(self.frame_name, self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

if __name__ == '__main__':

    src_list = [0,1]
    cam_widgets = []
    for src in src_list:
        cam_widgets.append(VideoStreamWidget(src))

    while True:
        try:
            for cam in cam_widgets:
                cam.show_frame()
            # cam1.show_frame()
        except AttributeError:
            pass