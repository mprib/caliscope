import sys
from pathlib import Path
import time

import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap, QFont

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FrameEmitter(QThread):
    # establish signals from the frame that will be displayed in real time 
    # within the GUI
    ImageBroadcast = pyqtSignal(QImage)
    FPSBroadcast = pyqtSignal(int)

    
    def __init__(self, camcap, pixmap_edge_length=None):
        # pixmap_edge length is from the display window. It will rescale the window
        # to always have square dimensions with black around either side
        super(FrameEmitter,self).__init__()
        self.min_sleep = .01 # if true fps drops to zero, don't blow up
        self.camcap = camcap
        print("Initializing Frame Emitter")
    
    def run(self):
        MIN_SLEEP_TIME = .01
        self.ThreadActive = True
        self.height = int(self.camcap.cam.resolution[0])
        self.width = int(self.camcap.cam.resolution[1])
         
        while self.ThreadActive:
            try:    # takes a moment for capture widget to spin up...don't error out

                # Grab a frame from the capture widget and broadcast to displays
                frame = self.camcap.frame
                Pic = self.cv2_to_qlabel(frame)
                self.ImageBroadcast.emit(Pic)

                # grab and broadcast fps
                fps = self.camcap.FPS_actual
                self.FPSBroadcast.emit(fps)

                # throttle rate of broadcast to reduce system overhead
                if fps == 0:    # Camera likely reconnecting
                    time.sleep(MIN_SLEEP_TIME) 
                else:
                    time.sleep(1/fps)

            except AttributeError:
                pass

    def stop(self):
        self.ThreadActive = False
        self.quit()

    def cv2_to_qlabel(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FlippedImage = cv2.flip(Image, 1)

        qt_frame = QImage(FlippedImage.data, 
                          FlippedImage.shape[1], 
                          FlippedImage.shape[0], 
                          QImage.Format.Format_RGB888)
        return qt_frame
    
     

if __name__ == "__main__":
    pass

    # not much to look at here... go to camera_config_widget.py for test