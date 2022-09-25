from threading import Thread
import cv2

class RTSPVideoWriterObject(object):
    def __init__(self, src):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)
        self.src = src
        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        
        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.output_video = cv2.VideoWriter(str(src) + '_output.avi', self.codec, 30, (self.frame_width, self.frame_height))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow(str(self.src), self.frame)
        
        # Press ESC on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == 27: 
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit()

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)

if __name__ == '__main__':
    # rtsp_stream_link = 'your stream link!'
    video_stream_widgets = []
    for src in [0,1]:    
        video_stream_widgets.append(RTSPVideoWriterObject(src))
    while True:
        try:
            for widget in video_stream_widgets:
                widget.show_frame()
                widget.save_frame()
        except AttributeError:
            pass
