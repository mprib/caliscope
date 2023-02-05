import calicam.logger
logger = calicam.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QSpinBox,
    QComboBox,
    QCheckBox,
    QDialog,
    QGroupBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

# Append main repo to top of path to allow import of backend
from calicam.gui_wizard.wizard_camera_config.wizard_frame_emitter import FrameEmitter

from calicam.session import Session


class CameraConfigDialog(QDialog):
    def __init__(self, session, port):
        super(CameraConfigDialog, self).__init__()

        # set up variables for ease of reference
        self.session = session
        # self.monocal = session.monocalibrators[port]
        self.port = port
        self.stream = self.session.streams[self.port]
        # self.stream.camera = self.stream.camera

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
        controls = QHBoxLayout()
        # controls.addWidget(self.fps_grp)
        # controls.addWidget(self.grid_grp)
        self.v_box.addWidget(self.fps_grp)
        # self.v_box.addWidget(self.grid_grp)
        self.v_box.addLayout(controls)

        ###################### CALIBRATION  ################################
        # self.v_box.addWidget(self.calibrate_grp)

    ####################### SUB_WIDGET CONSTRUCTION ###############################

    def build_calibrate_grp(self):
        logger.debug("Building Calibrate Group")
        self.calibrate_grp = QGroupBox("Calibrate")
        # Generally Horizontal Configuration
        hbox = QHBoxLayout()
        self.calibrate_grp.setLayout(hbox)

        # Collect Calibration Corners
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        collect_crnr_btn = QPushButton("Capture")
        collect_crnr_btn.setMaximumWidth(100)
        vbox.addWidget(collect_crnr_btn)

        def capture():
            """change to turn on/off"""
            if not self.monocal.capture_corners:
                self.monocal.capture_corners = True
                collect_crnr_btn.setText("Stop Capture")
                self.calibrate_btn.setEnabled(False)
            else:
                self.monocal.capture_corners = False
                collect_crnr_btn.setText("Capture")
                if self.monocal.grid_count > 1:
                    self.calibrate_btn.setEnabled(True)
                    self.clear_grid_history_btn.setEnabled(True)

        collect_crnr_btn.clicked.connect(capture)

        # Calibrate Button
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setMaximumWidth(100)
        vbox.addWidget(self.calibrate_btn)

        def calibrate():
            if len(self.monocal.all_ids) > 0:
                self.cal_output.setText("Calibration can take a moment...")
                self.calibrate_btn.setEnabled(False)
                self.clear_grid_history_btn.setEnabled(True)
                self.save_cal_btn.setEnabled(True)
                self.undistort_btn.setEnabled(True)

                def wrker():
                    self.monocal.calibrate()
                    self.cal_output.setText(self.stream.camera.calibration_summary())

                self.calib_thread = Thread(target=wrker, args=(), daemon=True)
                self.calib_thread.start()
            else:
                self.cal_output.setText("Need to Collect Grids")

        # self.calibrate_btn.clicked.connect(calibrate)

        # Clear calibration history
        self.clear_grid_history_btn = QPushButton("Clear History")
        self.clear_grid_history_btn.setMaximumWidth(100)
        self.clear_grid_history_btn.setEnabled(False)
        vbox.addWidget(self.clear_grid_history_btn)

        def clear_capture_history():
            self.monocal.initialize_grid_history()
            self.calibrate_btn.setEnabled(False)
            self.clear_grid_history_btn.setEnabled(False)
            # self.save_cal_btn.setEnabled(False)
            self.undistort_btn.setEnabled(False)
            self.frame_emitter.undistort = False

        self.clear_grid_history_btn.clicked.connect(clear_capture_history)

        # Undistort
        self.undistort_btn = QPushButton("Undistort")

        # check here to see if distortion params are available for this camera
        if self.stream.camera.distortion is None:
            self.undistort_btn.setEnabled(False)
        else:
            self.undistort_btn.setEnabled(True)

        self.undistort_btn.setMaximumWidth(100)
        vbox.addWidget(self.undistort_btn)

        def undistort():
            if self.frame_emitter.undistort:
                self.frame_emitter.undistort = False
                self.undistort_btn.setText("Undistort")
            else:
                self.frame_emitter.undistort = True
                self.undistort_btn.setText("Revert")

        self.undistort_btn.clicked.connect(undistort)

        def on_save_click():
            self.session.save_camera(self.port)

        # Save Calibration
        self.save_cal_btn = QPushButton("Save Calibration")
        # self.save_cal_btn.setEnabled(False)
        self.save_cal_btn.setMaximumWidth(100)
        self.save_cal_btn.clicked.connect(on_save_click)
        vbox.addWidget(self.save_cal_btn)

        # include calibration grid in horizontal box
        hbox.addLayout(vbox)

        self.cal_output = QLabel()
        self.cal_output.setWordWrap(True)
        self.cal_output.setMaximumWidth(int(self.pixmap_edge / 3))
        self.cal_output.setText(self.stream.camera.calibration_summary())
        hbox.addWidget(self.cal_output)
        # calib_output.setMaximumWidth()

    def build_fps_grp(self):

        self.fps_grp = QGroupBox("FPS")
        fps_hbox = QHBoxLayout()
        self.fps_grp.setLayout(fps_hbox)

        logger.debug("Building FPS Control")
        fps_hbox.addWidget(QLabel("Target:"))
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(self.stream.fps)


        def on_frame_rate_spin(fps_rate):
            self.stream.set_fps_target(fps_rate)
            logger.info(f"Changing stream frame rate for port{self.port}")

        self.frame_rate_spin.valueChanged.connect(on_frame_rate_spin)
        fps_hbox.addWidget(self.frame_rate_spin)

        self.fps_display = QLabel()
        self.fps_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        fps_hbox.addWidget(self.fps_display)

        def FPSUpdateSlot(fps):
            if self.stream.camera.capture.isOpened():
                # rounding to nearest integer should be close enough for our purposes
                self.fps_display.setText("Actual: " + str(round(fps, 1)))
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

    def build_ccw_rotation_btn(self):
        self.ccw_rotation_btn = QPushButton(
            QIcon("calicam/gui/icons/rotate-camera-left.svg"), ""
        )
        self.ccw_rotation_btn.setMaximumSize(100, 50)

        # Clockwise rotation called because the display image is flipped
        self.ccw_rotation_btn.clicked.connect(self.stream.camera.rotate_CW)

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

        self.exposure_hbox.addWidget(label)
        self.exposure_hbox.addWidget(self.exp_slider)
        self.exposure_hbox.addWidget(exp_number)

        self.exposure_hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def build_ignore_checkbox(self):

        self.ignore_box = QCheckBox("Ignore", self)
        self.exposure_hbox.addWidget(self.ignore_box)

        def ignore_cam(signal):
            print(signal)
            if signal == 0:  # not checked
                logger.info(f"Don't ignore camera at port {self.port}")
                self.stream.camera.ignore = False
            else:  # value of checkState() might be 2?
                logger.info(f"Ignore camera at port {self.port}")
                self.stream.camera.ignore = True

        self.ignore_box.stateChanged.connect(ignore_cam)

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
            for w, h in self.stream.camera.possible_resolutions:
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

            # whenever resolution changes, calibration parameters no longer apply
            self.stream.camera.error = None
            self.stream.camera.camera_matrix = None
            self.stream.camera.distortion = None
            self.stream.camera.grid_count = 0
            self.frame_emitter.undistort = False

            self.cal_output.setText(self.stream.camera.calibration_summary())
            self.clear_grid_history_btn.click()

        self.resolution_combo = QComboBox()

        self.resolution_combo.addItems(resolutions_text())
        self.resolution_combo.setMaximumSize(100, 50)

        w, h = self.stream.camera.resolution
        self.resolution_combo.setCurrentText(f"{int(w)} x {int(h)}")
        self.resolution_combo.currentTextChanged.connect(change_resolution)


if __name__ == "__main__":
    App = QApplication(sys.argv)

    repo = Path(str(Path(__file__)).split("calicam")[0],"calicam")
    config_path = Path(repo, "sessions", "default_res_session")

    # THIS IS WHERE YOU START TOMORROW, MAC. MAKE THIS WORK WITH JUST monocalibrator INPUT
    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    # session.adjust_resolutions()
    # session.load_monocalibrators()

    test_port = 0

    logger.info("Creating Camera Config Dialog")
    cam_dialog = CameraConfigDialog(session, test_port)

    logger.info("About to show camera config dialog")
    cam_dialog.show()

    sys.exit(App.exec())
