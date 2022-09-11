import cv2 as cv
import time 

class CameraFeeds():
    """
    Create a set of live OpenCV videoCapture devices that will be able to 
    generate calibration parameters. 
    """

    def __init__(self, input_streams,stream_names):
        "Initialize camera object"
        self.input_streams = input_streams
        self.stream_names = stream_names

    def open_captures(self):
        self.captures = {}

        for nm, strm in zip(self.stream_names, self.input_streams):
            self.captures[nm] = cv.VideoCapture(strm)

        while True:
            self.frames = {}

            for nm, cap in self.captures.items():
                _, self.frames[nm] = cap.read()

                cv.imshow(nm, self.frames[nm])

            if cv.waitKey(5) == 27: # ESC to stop   
                break

        self.destroy_captures()

    def destroy_captures(self):
        for nm, cap in self.captures.items():
            cap.release()
        cv.destroyAllWindows()

    def show(self):
        self.capture = cv.VideoCapture(self.input_stream)

        while self.capture.isOpened():

            success, frame = self.capture.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)




if __name__ == "__main__":
    feeds = CameraFeeds([0,1], ["Cam_1", "Cam_2"])
    feeds.open_captures()