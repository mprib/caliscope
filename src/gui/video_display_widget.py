# At this point I can't even recall where I copied this code from, but it is what
# actually works and drives home to me that I need to develop a better understanding
# of threads, signals, slots, events, all that as it relates to GUI development

# Moving on to pythonguis.com tutorials for just that. But this code works...


#%%
from ast import arg
import sys
from pathlib import Path
import time
from threading import Thread

import cv2

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton,
                            QSlider, QComboBox, QDialog, QSizePolicy, QLCDNumber,
                            QToolBar, QLabel, QLineEdit, QCheckBox, QScrollArea,
                            QVBoxLayout, QHBoxLayout, QGridLayout)

from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap, QFont

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera_capture_widget import CameraCaptureWidget
from src.cameras.camera import Camera

class CameraConfigWidget(QDialog):
    def __init__(self, camcap):
        super(CameraConfigWidget, self).__init__()


        # frame emitter is a thread that is constantly pulling in values from 
        # the capture widget and broadcasting them to widgets on this window 
        self.cam_cap = camcap
        self.frame_emitter = FrameEmitter(self.cam_cap)
        self.frame_emitter.start()


        VBL = QVBoxLayout()
        self.setLayout(VBL)


        # print("About to add frame display")
        #################      VIDEO AT TOP     ##########################     
        VBL.addWidget(self.get_frame_display())
        VBL.setAlignment(Qt.AlignmentFlag.AlignTop)
        VBL.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        #######################     FPS         ##########################
        VBL.addWidget(self.get_fps_display())

        #############################  ADD HBOX ###########################
        HBL = QHBoxLayout()
        ### MP TOGGLE #####################################################
        HBL.addWidget(self.get_mediapipe_toggle())
        
        ################ ROTATE CCW #######################################
        HBL.addWidget(self.get_ccw_rotation_button())

        ############################## ROTATE CW ###########################
        HBL.addWidget(self.get_cw_rotation_button())
        # VBL.addWidget(self.mediapipeLabel)
        ######################################### RESOLUTION DROPDOWN ######
        HBL.addWidget(self.get_resolution_dropdown())
        HBL.setAlignment(Qt.AlignmentFlag.AlignCenter)

        VBL.addLayout(HBL)

        #################### EXPOSURE SLIDER ##############################
        VBL.addLayout(self.get_exposure_slider())


####################### SUB_WIDGET CONSTRUCTION ###############################
    def get_fps_display(self):

        fps_display = QLabel()
        fps_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        fps_display.setFont(QFont("Times New Roman", 15))

        def FPSUpdateSlot(fps):
            fps_display.setText(str(str(fps) + " FPS"))

        self.frame_emitter.FPSBroadcast.connect(FPSUpdateSlot)        
        
        return fps_display
 
    def get_cw_rotation_button(self):
        rotate_cw_btn = QPushButton("Rotate CW")
        rotate_cw_btn.setMaximumSize(100, 50)

        def rotate_cw():
            # Counter Clockwise rotation called because the display image is flipped
            self.cam_cap.rotate_CCW()

        rotate_cw_btn.clicked.connect(rotate_cw)

        return rotate_cw_btn

    def get_ccw_rotation_button(self):
        rotate_ccw_btn = QPushButton("Rotate CCW")
        rotate_ccw_btn.setMaximumSize(100, 50)

        def rotate_ccw():
            # Clockwise rotation called because the display image is flipped
            self.cam_cap.rotate_CW()

        rotate_ccw_btn.clicked.connect(rotate_ccw)

        return rotate_ccw_btn


    
    def get_exposure_slider(self):
        # construct a horizontal widget with label: slider: value display
        HBox = QHBoxLayout()
        label = QLabel("Exposure")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)

        exp_slider = QSlider(Qt.Orientation.Horizontal)
        exp_slider.setRange(-10,0)
        exp_slider.setSliderPosition(int(self.cam_cap.cam.exposure))
        exp_slider.setPageStep(1)
        exp_slider.setSingleStep(1)
        exp_slider.setMaximumWidth(400)
        exp_number = QLabel()
        exp_number.setText(str(int(self.cam_cap.cam.exposure)))

        def update_exposure(s):
            print(f"Exposure is {s}")
            self.cam_cap.cam.exposure = s
            exp_number.setText(str(s))

        exp_slider.valueChanged.connect(update_exposure)

        HBox.addWidget(label)
        HBox.addWidget(exp_slider)
        HBox.addWidget(exp_number)

        HBox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        return HBox


    def get_frame_display(self):
        # return a QLabel that is linked to the constantly changing image
        # emitted by the FrameEmitter thread (which is pulled off of the 
        # video capture widget which is running its own roll_camera thread)
        
        # IMPORTANT: frame_emitter thread must continue to exist after running
        # this method. Cannot be confined to namespace of the method
        CameraDisplay = QLabel()

        # I believe that since this is a text label (technically) it needs
        # to be centered itself. Centering the containing widget alignment 
        # won't do
        CameraDisplay.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        def ImageUpdateSlot(Image):
            CameraDisplay.setPixmap(QPixmap.fromImage(Image))

        self.frame_emitter.ImageBroadcast.connect(ImageUpdateSlot)

        return CameraDisplay

    def get_mediapipe_toggle(self):
        # Mediapip display toggle
        mediapipe_toggle = QCheckBox("Show Mediapipe Overlay")
        mediapipe_toggle.setCheckState(Qt.CheckState.Checked)

        def toggle_mediapipe(s):
            print("Toggle Mediapipe")
            self.cam_cap.toggle_mediapipe()

        mediapipe_toggle.stateChanged.connect(toggle_mediapipe)

        return mediapipe_toggle

    def get_resolution_dropdown(self):

        # possible resolutions is a list of tuples, but we need text
        def resolutions_text():
            res_text = []
            for w, h in self.cam_cap.cam.possible_resolutions:
                res_text.append(f"{int(w)} x {int(h)}")
            return res_text
        
        def change_resolution(res_text):
            w, h = res_text.split("x")
            new_res = (int(w), int(h))

            self.change_res_thread = Thread(target = self.cam_cap.change_resolution,
                                            args = (new_res, ),
                                            daemon=True)
            self.change_res_thread.start()
            # self.cam_cap.change_resolution(new_res)

            self.setFixedSize(QSize(int(w) + 50, int(h)+50))

        resolution_combo = QComboBox()
        
        w,h = self.cam_cap.cam.default_resolution
        resolution_combo.setCurrentText(f"{int(w)} x {int(h)}")
        resolution_combo.setMaximumSize(100, 50)
        resolution_combo.addItems(resolutions_text())
        resolution_combo.currentTextChanged.connect(change_resolution)        
        return resolution_combo
        

class FrameEmitter(QThread):
    # establish signals from the frame that will be displayed in real time 
    # within the GUI
    ImageBroadcast = pyqtSignal(QImage)
    FPSBroadcast = pyqtSignal(int)
   
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
                # print(f"Flipped Image Shape is: {FlippedImage.shape}")

                qt_frame = QImage(FlippedImage.data, 
                                  FlippedImage.shape[1], 
                                  FlippedImage.shape[0], 
                                  QImage.Format.Format_RGB888)
                # Pic = qt_frame.scaled(self.width, self.height, Qt.AspectRatioMode.KeepAspectRatio)
                Pic = qt_frame
                self.ImageBroadcast.emit(Pic)
                self.FPSBroadcast.emit(fps)
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

