
#%%
import sys
from pathlib import Path
from threading import Thread
import time

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton,
                            QSlider, QComboBox, QDialog, QSizePolicy, QLCDNumber,
                            QToolBar, QLabel, QLineEdit, QCheckBox, QScrollArea,
                            QVBoxLayout, QHBoxLayout, QGridLayout, )

from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap, QFont

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera
from src.cameras.real_time_device import RealTimeDevice
from frame_emitter import FrameEmitter

class CameraConfigWidget(QDialog):
    def __init__(self, real_time_device, frame_emitter=None):
        super(CameraConfigWidget, self).__init__()
        # frame emitter is a thread that is constantly pulling in values from 
        # the capture widget and broadcasting them to widgets on this window 
        
        # print(self.isAnimated()) 
        # self.setAnimated(False) 
    
        self.RTD = real_time_device
        if frame_emitter:
            self.frame_emitter = frame_emitter
        else:
            self.frame_emitter = FrameEmitter(self.RTD)
            self.frame_emitter.start()

        self.setFixedSize(DISPLAY_HEIGHT/3, DISPLAY_HEIGHT/3)
        

        ################### BUILD SUB WIDGETS #############################
        self.build_frame_display()
        self.build_fps_display()
        self.build_mediapipe_toggle()
        self.build_ccw_rotation_btn()
        self.build_cw_rotation_btn()
        self.build_resolution_combo()
        self.build_exposure_hbox()
        ###################################################################
        self.VBL = QVBoxLayout()
        self.setLayout(self.VBL)
        # container = QWidget()
        # container.setLayout(self.VBL)
        # self.setCentralWidget(container)
        # VBL.setAlignment(Qt.AlignmentFlag.AlignTop)
        # VBL.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        #################      VIDEO AT TOP     ##########################     
        self.VBL.addWidget(self.frame_display)
        #######################     FPS         ##########################
        self.VBL.addWidget(self.fps_display)

        #############################  ADD HBOX ###########################
        HBL = QHBoxLayout()
        ### MP TOGGLE #####################################################
        HBL.addWidget(self.mediapipe_toggle)
        
        ################ ROTATE CCW #######################################
        HBL.addWidget(self.ccw_rotation_btn)

        ############################## ROTATE CW ###########################
        HBL.addWidget(self.cw_rotation_btn)
        # VBL.addWidget(self.mediapipeLabel)
        ######################################### RESOLUTION DROPDOWN ######
        HBL.addWidget(self.resolution_combo)
        HBL.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.VBL.addLayout(HBL)

        #################### EXPOSURE SLIDER ##############################
        self.VBL.addLayout(self.exposure_hbox)
        self.adjustSize()
####################### SUB_WIDGET CONSTRUCTION ###############################
    def build_fps_display(self):

        self.fps_display = QLabel()
        self.fps_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.fps_display.setFont(QFont("", 15))
        # self.fps_display.setMaximumSize(400,400)
        def FPSUpdateSlot(fps):
            if fps == 0:
                self.fps_display.setText("reconnecting to camera...")
            else:
                self.fps_display.setText(str(fps) + " FPS")

        self.frame_emitter.FPSBroadcast.connect(FPSUpdateSlot)        
 
    def build_cw_rotation_btn(self):
        self.cw_rotation_btn = QPushButton("Rotate CW")
        self.cw_rotation_btn.setMaximumSize(100, 50)

        def rotate_cw():
            # Counter Clockwise rotation called because the display image is flipped
            self.VBL.removeWidget(self.frame_display)
            self.frame_display.close()
            self.RTD.rotate_CCW()
            self.build_frame_display()
            self.VBL.insertWidget(0, self.frame_display)

        self.cw_rotation_btn.clicked.connect(rotate_cw)

    def build_ccw_rotation_btn(self):
        self.ccw_rotation_btn = QPushButton("Rotate CCW")
        self.ccw_rotation_btn.setMaximumSize(100, 50)

        def rotate_ccw():
            # Clockwise rotation called because the display image is flipped
            self.RTD.rotate_CW()
            # self.resize()

        self.ccw_rotation_btn.clicked.connect(rotate_ccw)
    
    def build_exposure_hbox(self):
        # construct a horizontal widget with label: slider: value display
        self.exposure_hbox = QHBoxLayout()
        label = QLabel("Exposure")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)

        exp_slider = QSlider(Qt.Orientation.Horizontal)
        exp_slider.setRange(-10,0)
        exp_slider.setSliderPosition(int(self.RTD.cam.exposure))
        exp_slider.setPageStep(1)
        exp_slider.setSingleStep(1)
        # exp_slider.setMaximumWidth(400)
        exp_number = QLabel()
        exp_number.setText(str(int(self.RTD.cam.exposure)))

        def update_exposure(s):
            self.RTD.cam.exposure = s
            exp_number.setText(str(s))

        exp_slider.valueChanged.connect(update_exposure)

        self.exposure_hbox.addWidget(label)
        self.exposure_hbox.addWidget(exp_slider)
        self.exposure_hbox.addWidget(exp_number)

        self.exposure_hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def build_frame_display(self):
        # return a QLabel that is linked to the constantly changing image
        # IMPORTANT: frame_emitter thread must continue to exist after running
        # this method. Cannot be confined to namespace of the method

        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setFixedWidth(self.width()-0.1*self.width())
        self.frame_display.setFixedHeight(self.height()-0.1*self.height())

        def ImageUpdateSlot(Image):
            pixmap = QPixmap.fromImage(Image)
            scaled_pixmap = pixmap.scaled(500,
                                          500,
                                          Qt.AspectRatioMode.KeepAspectRatio)

            self.frame_display.setPixmap(scaled_pixmap)

        self.frame_emitter.ImageBroadcast.connect(ImageUpdateSlot)

    def build_mediapipe_toggle(self):
        # Mediapip display toggle
        self.mediapipe_toggle = QCheckBox("Show Mediapipe Overlay")
        self.mediapipe_toggle.setCheckState(Qt.CheckState.Checked)

        def toggle_mediapipe(s):
            print("Toggle Mediapipe")
            self.RTD.toggle_mediapipe()

        self.mediapipe_toggle.stateChanged.connect(toggle_mediapipe)

        
    def build_resolution_combo(self):
        # possible resolutions is a list of tuples, but we need a list of Stext
        def resolutions_text():
            res_text = []
            for w, h in self.RTD.cam.possible_resolutions:
                res_text.append(f"{int(w)} x {int(h)}")
            return res_text
        
        def change_resolution(res_text):
            # call the cam_cap widget to change the resolution, but do it in a 
            # thread so that it doesn't halt your progress
            w, h = res_text.split("x")
            w, h = int(w), int(h)
            new_res = (w, h)
            # self.cam_cap.change_resolution(new_res)
            self.change_res_thread = Thread(target = self.RTD.change_resolution,
                                            args = (new_res, ),
                                            daemon=True)
            self.change_res_thread.start()

        
        self.resolution_combo = QComboBox()
        
        w,h = self.RTD.cam.default_resolution
        
        self.resolution_combo.setCurrentText(f"{int(w)} x {int(h)}")
        self.resolution_combo.setMaximumSize(100, 50)
        self.resolution_combo.addItems(resolutions_text())
        self.resolution_combo.currentTextChanged.connect(change_resolution)        



if __name__ == "__main__":
    port = 0
    cam = Camera(port)
 
    App = QApplication(sys.argv)
    DISPLAY_WIDTH = App.primaryScreen().size().width()
    DISPLAY_HEIGHT = App.primaryScreen().size().height()
    real_time_device = RealTimeDevice(cam)
    display = CameraConfigWidget(real_time_device)
    display.show()
    sys.exit(App.exec())

