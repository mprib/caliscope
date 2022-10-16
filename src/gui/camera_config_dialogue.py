
#%%
from re import I
import sys
from pathlib import Path
from threading import Thread
import time
from tkinter import W
import cv2

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton,
                            QSlider, QComboBox, QDialog, QSizePolicy, QLCDNumber,
                            QToolBar, QLabel, QLineEdit, QCheckBox, QScrollArea,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QRadioButton)

from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap, QFont

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera
from src.cameras.real_time_device import RealTimeDevice
from frame_emitter import FrameEmitter

class CameraConfigDialog(QDialog):
    def __init__(self, real_time_device, frame_emitter=None):
        super(CameraConfigDialog, self).__init__()
        # frame emitter is a thread that is constantly pulling in values from 
        # the capture widget and broadcasting them to widgets on this window 
        
        # print(self.isAnimated()) 
        # self.setAnimated(False) 
        App = QApplication.instance()
        DISPLAY_WIDTH = App.primaryScreen().size().width()
        DISPLAY_HEIGHT = App.primaryScreen().size().height()

        self.RTD = real_time_device


        pixmap_edge = min(DISPLAY_WIDTH/2, DISPLAY_HEIGHT/2)
        self.frame_emitter = FrameEmitter(self.RTD, pixmap_edge)
        self.frame_emitter.start()
        self.setFixedSize(pixmap_edge, pixmap_edge + 200) 
        self.setContentsMargins(0,0,0,0)

        ################### BUILD SUB WIDGETS #############################
        self.build_frame_display()
        self.build_fps_display()
        self.build_ccw_rotation_btn()
        self.build_cw_rotation_btn()
        self.build_resolution_combo()
        self.build_exposure_hbox()
        self.build_view_full_res_btn()
        self.build_toggle_grp()
        ###################################################################
        self.VBL = QVBoxLayout(self)
        self.VBL.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.VBL.setContentsMargins(0,0,0,0)

        #################      VIDEO AT TOP     ##########################     
        self.VBL.addWidget(self.frame_display)

        ############################  ADD HBOX OF CONFIG ######################
        HBL = QHBoxLayout()
        
        ################ ROTATE CCW #######################################
        HBL.addWidget(self.ccw_rotation_btn)

        ############################## ROTATE CW ###########################
        HBL.addWidget(self.cw_rotation_btn)
        # VBL.addWidget(self.mediapipeLabel)
        ######################################### RESOLUTION DROPDOWN ######
        HBL.addWidget(self.resolution_combo)
        HBL.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.VBL.addLayout(HBL)


        #################### EXPOSURE SLIDER #################################
        self.VBL.addLayout(self.exposure_hbox)
        
        #######################     FPS         ##############################
        self.VBL.addWidget(self.fps_display)
        ###################### RADIO BUTTONS OF OVERLAY TOGGLES ##################
        self.VBL.addWidget(self.toggle_grp)

        ################## FULL RESOLUTION LAUNCH BUTTON ######################
        self.VBL.addWidget(self.view_full_res_btn)

####################### SUB_WIDGET CONSTRUCTION ###############################
    def build_toggle_grp(self):  

        def on_radio_btn():
            radio_grp = self.sender().text()
            if radio_grp == "None":
                self.RTD.show_mediapipe = False
            if radio_grp == "Mediapipe Hands":
                self.RTD.show_mediapipe = True


        self.toggle_grp = QGroupBox("Toggle Visual Overlays to Confirm Capture Quality")
        # self.toggle_grp.setFixedWidth(0.75* self.width-50())
        hbox = QHBoxLayout()
        for option in ["None", "Mediapipe Hands", "Charuco"]:
            btn = QRadioButton(option)
            hbox.addWidget(btn)
            if option == "None":
                btn.setChecked(True)
            btn.toggled.connect(on_radio_btn)
        
        self.toggle_grp.setLayout(hbox)
        hbox.setAlignment(Qt.AlignmentFlag.AlignHCenter)


    def build_fps_display(self):

        self.fps_display = QLabel()
        self.fps_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        # self.fps_display.setFont(QFont("", 15))
        # self.fps_display.setMaximumSize(400,400)
        def FPSUpdateSlot(fps):
            if fps == 0:
                self.fps_display.setText("reconnecting to camera...")
            else:
                self.fps_display.setText("FPS: " + str(fps))

        self.frame_emitter.FPSBroadcast.connect(FPSUpdateSlot)        
 
    def build_cw_rotation_btn(self):
        self.cw_rotation_btn = QPushButton("Rotate CW")
        self.cw_rotation_btn.setMaximumSize(100, 50)

        def rotate_cw():
            # Counter Clockwise rotation called because the display image is flipped
            self.RTD.rotate_CCW()
            self.adjustSize()

        self.cw_rotation_btn.clicked.connect(rotate_cw)

    def build_ccw_rotation_btn(self):
        self.ccw_rotation_btn = QPushButton("Rotate CCW")
        self.ccw_rotation_btn.setMaximumSize(100, 50)

        def rotate_ccw():
            # Clockwise rotation called because the display image is flipped
            self.RTD.rotate_CW()
            self.adjustSize()

        self.ccw_rotation_btn.clicked.connect(rotate_ccw)
    
    def build_exposure_hbox(self):
        # construct a horizontal widget with label: slider: value display
        self.exposure_hbox = QHBoxLayout()
        label = QLabel("Exposure")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-10,0)
        self.exp_slider.setSliderPosition(int(self.RTD.cam.exposure))
        self.exp_slider.setPageStep(1)
        self.exp_slider.setSingleStep(1)
        self.exp_slider.setMaximumWidth(200)
        exp_number = QLabel()
        exp_number.setText(str(int(self.RTD.cam.exposure)))

        def update_exposure(s):
            self.RTD.cam.exposure = s
            exp_number.setText(str(s))

        self.exp_slider.valueChanged.connect(update_exposure)

        self.exposure_hbox.addWidget(label)
        self.exposure_hbox.addWidget(self.exp_slider)
        self.exposure_hbox.addWidget(exp_number)

        self.exposure_hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def build_frame_display(self):
        # return a QLabel that is linked to the constantly changing image
        # IMPORTANT: frame_emitter thread must continue to exist after running
        # this method. Cannot be confined to namespace of the method

        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setFixedWidth(self.width())
        self.frame_display.setFixedHeight(self.width())
        w = self.frame_display.width()
        h = self.frame_display.height()
        def ImageUpdateSlot(QPixmap):

            self.frame_display.setPixmap(QPixmap)

        self.frame_emitter.ImageBroadcast.connect(ImageUpdateSlot)


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

    def build_view_full_res_btn(self):
        self.view_full_res_btn = QPushButton("Open Full Resolution Window (press 'q' to close)")
        
        def cv2_view_worker():
            while True:
                frame = cv2.flip(self.RTD.frame, 1)

                cv2.imshow("Press 'q' to Quit", frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break

        def run_cv2_view():
            self.cv2_view = Thread(target=cv2_view_worker, args = (), daemon = True)
            self.cv2_view.start()

        self.view_full_res_btn.clicked.connect(run_cv2_view)

if __name__ == "__main__":
    port = 0
    cam = Camera(port)
 
    App = QApplication(sys.argv)
    # DISPLAY_WIDTH = App.primaryScreen().size().width()
    # DISPLAY_HEIGHT = App.primaryScreen().size().height()
    real_time_device = RealTimeDevice(cam)
    display = CameraConfigDialog(real_time_device)
    display.show()
    sys.exit(App.exec())


# %%
