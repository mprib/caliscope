
#%%
import sys
from pathlib import Path
from threading import Thread

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton,
                            QSlider, QComboBox, QDialog, QSizePolicy, QLCDNumber,
                            QToolBar, QLabel, QLineEdit, QCheckBox, QScrollArea,
                            QVBoxLayout, QHBoxLayout, QGridLayout)

from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap, QFont

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera
from src.cameras.frame_capture_widget import CameraCaptureWidget
from frame_emitter import FrameEmitter

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
            fps_display.setText(str(fps) + " FPS")

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
        # IMPORTANT: frame_emitter thread must continue to exist after running
        # this method. Cannot be confined to namespace of the method
        CameraDisplay = QLabel()
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
        # possible resolutions is a list of tuples, but we need a list of Stext
        def resolutions_text():
            res_text = []
            for w, h in self.cam_cap.cam.possible_resolutions:
                res_text.append(f"{int(w)} x {int(h)}")
            return res_text
        
        def change_resolution(res_text):
            # call the cam_cap widget to change the resolution, but do it in a 
            # thread so that it doesn't halt your progress
            w, h = res_text.split("x")
            new_res = (int(w), int(h))
            self.change_res_thread = Thread(target = self.cam_cap.change_resolution,
                                            args = (new_res, ),
                                            daemon=True)
            self.change_res_thread.start()

            self.setFixedSize(QSize(int(w) + 50, int(h)+50))

        resolution_combo = QComboBox()
        
        w,h = self.cam_cap.cam.default_resolution
        resolution_combo.setCurrentText(f"{int(w)} x {int(h)}")
        resolution_combo.setMaximumSize(100, 50)
        resolution_combo.addItems(resolutions_text())
        resolution_combo.currentTextChanged.connect(change_resolution)        
        return resolution_combo
        


if __name__ == "__main__":
    port = 0
    cam = Camera(port)
    App = QApplication(sys.argv)
    camcap = CameraCaptureWidget(cam)
    display = CameraConfigWidget(camcap)
    display.show()
    sys.exit(App.exec())

