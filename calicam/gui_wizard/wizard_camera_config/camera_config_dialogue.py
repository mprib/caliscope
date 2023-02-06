import calicam.logger
logger = calicam.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QSpinBox,
    QComboBox,
    QCheckBox,
    QWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

# Append main repo to top of path to allow import of backend
from calicam.gui_wizard.wizard_camera_config.wizard_frame_emitter import FrameEmitter

from calicam.session import Session

FPS_TARGET = 100 # don't bother with fps at the moment. Just show at max actual fps

class CameraConfigDialog(QWidget):
    
    def __init__(self, session, port):
        super(CameraConfigDialog, self).__init__()

        # set up variables for ease of reference
        self.session = session
        self.port = port
        self.stream = self.session.streams[self.port]
        self.stream.set_fps_target(FPS_TARGET)
        
        App = QApplication.instance()
        DISPLAY_WIDTH = App.primaryScreen().size().width()
        DISPLAY_HEIGHT = App.primaryScreen().size().height()

        self.setWindowTitle("Camera Configuration and Calibration")

        self.pixmap_edge = min(DISPLAY_WIDTH / 3, DISPLAY_HEIGHT / 3)
        self.frame_emitter = FrameEmitter(self.stream, self.pixmap_edge)
        self.frame_emitter.start()
        self.setContentsMargins(0, 0, 0, 0)

        ################### BUILD SUB WIDGETS #############################
        self.build_frame_display()
        self.build_ccw_rotation_btn()
        self.build_cw_rotation_btn()
        self.build_resolution_combo()
        self.build_exposure_hbox()
        self.build_ignore_checkbox()
        self.build_fps_grp()
        ###################################################################
        self.v_box = QVBoxLayout(self)
        self.v_box.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.v_box.setContentsMargins(0, 0, 0, 0)

        #################      VIDEO AT TOP     ##########################
        self.v_box.addWidget(self.frame_display)

        ############################  ADD HBOX OF CONFIG ######################
        h_box = QHBoxLayout()
        h_box.addWidget(self.ccw_rotation_btn)
        h_box.addWidget(self.cw_rotation_btn)
        h_box.addWidget(self.resolution_combo)

        h_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.v_box.addLayout(h_box)

        #################### EXPOSURE SLIDER #################################
        self.v_box.addLayout(self.exposure_hbox)

        #######################     FPS   + Grid Count #########################
        self.other_controls = QHBoxLayout()
        self.other_controls.addLayout(self.fps_hbox)
        self.v_box.addLayout(self.other_controls)
        self.other_controls.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.v_box.addWidget(self.ignore_box)

    ####################### SUB_WIDGET CONSTRUCTION ###############################
    def save_camera(self):
        self.session.save_camera(self.port)
        
        
    def build_fps_grp(self):

        # self.fps_grp = QGroupBox("FPS")
        self.fps_hbox = QHBoxLayout()

        self.fps_display = QLabel()
        self.fps_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.fps_hbox.addWidget(self.fps_display)

        def FPSUpdateSlot(fps):
            if self.stream.camera.capture.isOpened():
                self.fps_display.setText("FPS: " + str(round(fps, 1)))
            else:
                self.fps_display.setText("reconnecting to camera...")

        self.frame_emitter.FPSBroadcast.connect(FPSUpdateSlot)


    def build_cw_rotation_btn(self):
        self.cw_rotation_btn = QPushButton(
            QIcon("calicam/gui/icons/rotate-camera-right.svg"), ""
        )
        self.cw_rotation_btn.setMaximumSize(100, 50)

        # Counter Clockwise rotation called because the display image is flipped
        self.cw_rotation_btn.clicked.connect(self.stream.camera.rotate_CCW)
        self.cw_rotation_btn.clicked.connect(self.save_camera)

    def build_ccw_rotation_btn(self):
        self.ccw_rotation_btn = QPushButton(
            QIcon("calicam/gui/icons/rotate-camera-left.svg"), ""
        )
        self.ccw_rotation_btn.setMaximumSize(100, 50)

        # Clockwise rotation called because the display image is flipped
        self.ccw_rotation_btn.clicked.connect(self.stream.camera.rotate_CW)
        self.ccw_rotation_btn.clicked.connect(self.save_camera)

    def build_exposure_hbox(self):
        # construct a horizontal widget with label: slider: value display
        self.exposure_hbox = QHBoxLayout()
        label = QLabel("Exposure")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-10, 0)
        self.exp_slider.setSliderPosition(int(self.stream.camera.exposure))
        self.exp_slider.setPageStep(1)
        self.exp_slider.setSingleStep(1)
        self.exp_slider.setMaximumWidth(200)
        exp_number = QLabel()
        exp_number.setText(str(int(self.stream.camera.exposure)))

        def update_exposure(s):
            self.stream.camera.exposure = s
            exp_number.setText(str(s))

        self.exp_slider.valueChanged.connect(update_exposure)
        self.exp_slider.valueChanged.connect(self.save_camera)

        self.exposure_hbox.addWidget(label)
        self.exposure_hbox.addWidget(self.exp_slider)
        self.exposure_hbox.addWidget(exp_number)

        self.exposure_hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def build_ignore_checkbox(self):

        self.ignore_box = QCheckBox("Ignore", self)
        def ignore_cam(signal):
            print(signal)
            if signal == 0:  # not checked
                logger.info(f"Don't ignore camera at port {self.port}")
                self.stream.camera.ignore = False
            else:  # value of checkState() might be 2?
                logger.info(f"Ignore camera at port {self.port}")
                self.stream.camera.ignore = True

        self.ignore_box.stateChanged.connect(ignore_cam)
        self.ignore_box.stateChanged.connect(self.save_camera)

    def build_frame_display(self):
        # return a QLabel that is linked to the constantly changing image

        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_display.setFixedWidth(self.width())
        self.frame_display.setFixedHeight(self.width())

        def ImageUpdateSlot(QPixmap):
            self.frame_display.setPixmap(QPixmap)

        self.frame_emitter.ImageBroadcast.connect(ImageUpdateSlot)

    def build_resolution_combo(self):
        def resolutions_text():
            res_text = []
            for w, h in self.stream.camera.verified_resolutions:
                res_text.append(f"{int(w)} x {int(h)}")
            return res_text

        def change_resolution(res_text):
            # call the cam_cap widget to change the resolution, but do it in a
            # thread so that it doesn't halt your progress

            w, h = res_text.split("x")
            w, h = int(w), int(h)
            new_res = (w, h)
            # self.cam_cap.change_resolution(new_res)
            logger.info(
                f"Attempting to change resolution of camera at port {self.port}"
            )
            self.change_res_thread = Thread(
                target=self.stream.change_resolution,
                args=(new_res,),
                daemon=True,
            )
            self.change_res_thread.start()

        self.resolution_combo = QComboBox()

        self.resolution_combo.addItems(resolutions_text())
        self.resolution_combo.setMaximumSize(100, 50)

        w, h = self.stream.camera.resolution
        self.resolution_combo.setCurrentText(f"{int(w)} x {int(h)}")
        self.resolution_combo.currentTextChanged.connect(change_resolution)
        self.resolution_combo.currentTextChanged.connect(self.save_camera)


if __name__ == "__main__":
    App = QApplication(sys.argv)

    repo = Path(str(Path(__file__)).split("calicam")[0],"calicam")
    config_path = Path(repo, "sessions", "default_res_session")

    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()

    test_port = 0

    logger.info("Creating Camera Config Dialog")
    cam_dialog = CameraConfigDialog(session, test_port)

    logger.info("About to show camera config dialog")
    cam_dialog.show()

    sys.exit(App.exec())
