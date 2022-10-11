# At this point I can't even recall where I copied this code from, but it is what
# actually works and drives home to me that I need to develop a better understanding
# of threads, signals, slots, events, all that as it relates to GUI development

# Moving on to pythonguis.com tutorials for just that. But this code works...


#%%
import sys
from pathlib import Path
import time
from threading import Thread

import cv2

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton,
                            QSlider,
                            QToolBar, QLabel, QLineEdit, QCheckBox, QScrollArea,
                            QVBoxLayout, QHBoxLayout, QGridLayout)

from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap


# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.video_capture_widget import CameraCaptureWidget
from src.cameras.camera import Camera

class CameraConfigWidget(QWidget):
    def __init__(self, camcap):
        super(CameraConfigWidget, self).__init__()

        self.cam_cap = camcap
        # Video display at top and settings beneath
        self.VBL = QVBoxLayout()
        self.HBL = QHBoxLayout()

        # print("About to add frame display")
        self.VBL.addWidget(self.get_frame_display())

        ################# BEGIN ADDING THE HBOX ###########################
        # print("about to add mp toggle")
        self.HBL.addWidget(self.get_mediapipe_toggle())
        
        # Image Rotation CCW
        self.rotate_ccw_btn = QPushButton("Rotate CCW")
        self.rotate_ccw_btn.clicked.connect(self.rotate_ccw)
        self.HBL.addWidget(self.rotate_ccw_btn)

        # Image Rotation CW 
        self.rotate_cw_btn = QPushButton("Rotate CW")
        self.rotate_cw_btn.clicked.connect(self.rotate_cw)
        self.HBL.addWidget(self.rotate_cw_btn)
        # self.VBL.addWidget(self.mediapipeLabel)

        # Horizontal Box with Exposure Slider Section
        # self.exposure_slider()


        self.setLayout(self.VBL)
        self.VBL.addLayout(self.HBL)
        self.VBL.addLayout(self.get_exposure_slider())

### Begin Exposure Setting
    def get_exposure_slider(self):
        # construct a horizontal widget with label: slider: value display
        HBox = QHBoxLayout()
        label = QLabel("Exposure")
        HBox.addWidget(label)

        exp_slider = QSlider(Qt.Orientation.Horizontal)
        exp_slider.setRange(-10,0)
        exp_slider.setSliderPosition(int(self.cam_cap.cam.exposure))
        exp_slider.setPageStep(1)
        exp_slider.setSingleStep(1)
        def update_exposure(s):
            print(f"Exposure is {s}")
            self.cam_cap.cam.exposure = s
        exp_slider.valueChanged.connect(update_exposure)

        HBox.addWidget(exp_slider)

        return HBox


    def get_frame_display(self):
        # return a QLabel that is linked to the constantly changing image
        # emitted by the FrameEmitter thread (which is pulled off of the 
        # video capture widget which is running its own roll_camera thread)
        
        # IMPORTANT: frame_emitter thread must continue to exist after running
        # this method. Cannot be confined to namespace of the method
        self.frame_emitter = FrameEmitter(self.cam_cap)
        self.frame_emitter.start()
        CameraDisplay = QLabel()

        def ImageUpdateSlot(Image):
            CameraDisplay.setPixmap(QPixmap.fromImage(Image))

        self.frame_emitter.ImageBroadcast.connect(ImageUpdateSlot)

        return CameraDisplay

    def get_mediapipe_toggle(self):
        # Mediapip display toggle
        self.mediapipe_toggle = QCheckBox("Show Mediapipe Overlay")
        self.mediapipe_toggle.setCheckState(Qt.CheckState.Checked)

        def toggle_mediapipe(s):
            print("Toggle Mediapipe")
            self.cam_cap.toggle_mediapipe()

        self.mediapipe_toggle.stateChanged.connect(toggle_mediapipe)

        return self.mediapipe_toggle

        
    def rotate_ccw(self):
        # Clockwise rotation called because the display image is flipped
        self.frame_emitter.camcap.rotate_CW()

    def rotate_cw(self):
        # Counter Clockwise rotation called because the display image is flipped
        self.frame_emitter.camcap.rotate_CCW()
            

    def CancelFeed(self):
        self.frame_emitter.stop()
        self.vid_cap_widget.capture.release()



class FrameEmitter(QThread):
    ImageBroadcast = pyqtSignal(QImage)
   
    def __init__(self, camcap):
        super(FrameEmitter,self).__init__()
        self.min_sleep = .01 # if true fps drops to zero, don't blow up
        self.camcap = camcap
        print("Initializing Frame Emitter")

    def run(self):
        # self.camcap = CameraCaptureWidget(self.camcap) #, self.width ,self.height)
        self.ThreadActive = True
        self.height = int(self.camcap.cam.resolution[0])
        self.width = int(self.camcap.cam.resolution[1])
         
        while self.ThreadActive:
            try:    # takes a moment for capture widget to spin up...don't error out

                # Grab a frame from the capture widget and adjust it to 
                frame = self.camcap.frame
                fps = self.camcap.FPS_actual
                self.fps_text =  str(int(round(fps, 0))) 
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)

                # overlay frame rate
                cv2.putText(FlippedImage, "FPS:" + self.fps_text, (10, 70),cv2.FONT_HERSHEY_TRIPLEX, 2,(0,165,255), 2)

                qt_frame = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format.Format_RGB888)
                Pic = qt_frame.scaled(self.width, self.height, Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageBroadcast.emit(Pic)
                time.sleep(1/fps)
                # time.sleep(1/self.peak_fps_display)

            except AttributeError:
                pass
    def stop(self):
        self.ThreadActive = False
        self.quit()

# if __name__ == "__main__":
# %%
port = 0
cam = Camera(port)
App = QApplication(sys.argv)
camcap = CameraCaptureWidget(cam)
display = CameraConfigWidget(camcap)
display.show()
sys.exit(App.exec())

