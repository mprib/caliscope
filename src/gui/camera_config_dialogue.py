import logging

LOG_FILE = "log/camera_config_dialog.log"
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
from pathlib import Path
from threading import Thread

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
)

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from frame_emitter import FrameEmitter

from src.calibration.charuco import Charuco
from src.cameras.camera import Camera
from src.cameras.video_stream import VideoStream
from src.session import Session


class CameraConfigDialog(QDialog):
    def __init__(self, session, port):
        super(CameraConfigDialog, self).__init__()

        self.session = session
        self.port = port

        self.monocal = session.monocalibrators[port]
        self.stream = session.streams[port]
        # stream reference needed to change resolution

        App = QApplication.instance()
        DISPLAY_WIDTH = App.primaryScreen().size().width()
        DISPLAY_HEIGHT = App.primaryScreen().size().height()

        self.setWindowTitle("Camera Configuration and Calibration")

        self.pixmap_edge = min(DISPLAY_WIDTH / 3, DISPLAY_HEIGHT / 3)
        self.frame_emitter = FrameEmitter(self.monocal, self.pixmap_edge)
        self.frame_emitter.start()
        self.setContentsMargins(0, 0, 0, 0)

        ################### BUILD SUB WIDGETS #############################
        self.build_frame_display()
        self.build_realtime_text_display()
        self.build_ccw_rotation_btn()
        self.build_cw_rotation_btn()
        self.build_resolution_combo()
        self.build_exposure_hbox()
        self.build_calibrate_grp()
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
        self.v_box.addLayout(self.realtime_text_hbox)

        ###################### CALIBRATION  ################################
        self.v_box.addWidget(self.calibrate_grp)

        for w in self.children():
            self.v_box.setAlignment(w, Qt.AlignmentFlag.AlignHCenter)

    ####################### SUB_WIDGET CONSTRUCTION ###############################

    def build_calibrate_grp(self):
        logging.debug("Building Calibrate Group")
        self.calibrate_grp = QGroupBox("Calibrate")
        # Generally Horizontal Configuration
        hbox = QHBoxLayout()
        self.calibrate_grp.setLayout(hbox)

        # Build Charuco Image Display
        self.charuco_display = QLabel()
        charuco_img = self.monocal.corner_tracker.charuco.board_pixmap(
            self.pixmap_edge / 3, self.pixmap_edge / 3
        )
        self.charuco_display.setPixmap(charuco_img)
        hbox.addWidget(self.charuco_display)

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
                    self.cal_output.setText(self.monocal.camera.calibration_summary())

                self.calib_thread = Thread(target=wrker, args=(), daemon=True)
                self.calib_thread.start()
            else:
                self.cal_output.setText("Need to Collect Grids")

        self.calibrate_btn.clicked.connect(calibrate)

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
        self.undistort_btn.setEnabled(False)
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

        # Save Calibration
        self.save_cal_btn = QPushButton("Save Calibration")
        # self.save_cal_btn.setEnabled(False)
        self.save_cal_btn.setMaximumWidth(100)
        vbox.addWidget(self.save_cal_btn)

        # TODO: refactor so saves managed elsewhere...why build a whole session
        # here just to save. There's got to be a more modular approach to this
        # that I expect will pay dividends later
        def save_cal():
            self.session.save_camera(self.port)

        self.save_cal_btn.clicked.connect(save_cal)

        # include calibration grid in horizontal box
        hbox.addLayout(vbox)

        self.cal_output = QLabel()
        self.cal_output.setWordWrap(True)
        self.cal_output.setMaximumWidth(self.pixmap_edge / 3)
        self.cal_output.setText(self.monocal.camera.calibration_summary())
        hbox.addWidget(self.cal_output)
        # calib_output.setMaximumWidth()

    def build_realtime_text_display(self):
        self.realtime_text_hbox = QHBoxLayout()
        logging.debug("Building FPS Display")
        self.fps_display = QLabel()
        self.fps_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.realtime_text_hbox.addWidget(self.fps_display)

        def FPSUpdateSlot(fps):
            if self.monocal.camera.is_rolling:
                # rounding to nearest integer should be close enough for our purposes
                self.fps_display.setText("FPS: " + str(round(fps, 0)))
            else:
                self.fps_display.setText("reconnecting to camera...")

        self.frame_emitter.FPSBroadcast.connect(FPSUpdateSlot)

        logging.debug("Building Grid Count Display")
        self.grid_count_display = QLabel()
        self.realtime_text_hbox.addWidget(self.grid_count_display)
        self.grid_count_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        def grid_count_update_slot(grid_count):
            self.grid_count_display.setText(f"Grid Count: {grid_count}")

        self.frame_emitter.GridCountBroadcast.connect(grid_count_update_slot)

    def build_cw_rotation_btn(self):
        self.cw_rotation_btn = QPushButton("Rotate CW")
        self.cw_rotation_btn.setMaximumSize(100, 50)

        # Counter Clockwise rotation called because the display image is flipped
        self.cw_rotation_btn.clicked.connect(self.monocal.camera.rotate_CCW)

    def build_ccw_rotation_btn(self):
        self.ccw_rotation_btn = QPushButton("Rotate CCW")
        self.ccw_rotation_btn.setMaximumSize(100, 50)

        # Clockwise rotation called because the display image is flipped
        self.ccw_rotation_btn.clicked.connect(self.monocal.camera.rotate_CW)

    def build_exposure_hbox(self):
        # construct a horizontal widget with label: slider: value display
        self.exposure_hbox = QHBoxLayout()
        label = QLabel("Exposure")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-10, 0)
        self.exp_slider.setSliderPosition(int(self.monocal.camera.exposure))
        self.exp_slider.setPageStep(1)
        self.exp_slider.setSingleStep(1)
        self.exp_slider.setMaximumWidth(200)
        exp_number = QLabel()
        exp_number.setText(str(int(self.monocal.camera.exposure)))

        def update_exposure(s):
            self.monocal.camera.exposure = s
            exp_number.setText(str(s))

        self.exp_slider.valueChanged.connect(update_exposure)

        self.exposure_hbox.addWidget(label)
        self.exposure_hbox.addWidget(self.exp_slider)
        self.exposure_hbox.addWidget(exp_number)

        self.exposure_hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def build_frame_display(self):
        # return a QLabel that is linked to the constantly changing image

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
        def resolutions_text():
            res_text = []
            for w, h in self.monocal.camera.possible_resolutions:
                res_text.append(f"{int(w)} x {int(h)}")
            return res_text

        def change_resolution(res_text):
            # call the cam_cap widget to change the resolution, but do it in a
            # thread so that it doesn't halt your progress

            w, h = res_text.split("x")
            w, h = int(w), int(h)
            new_res = (w, h)
            # self.cam_cap.change_resolution(new_res)
            self.change_res_thread = Thread(
                target=self.stream.change_resolution, args=(new_res,), daemon=True
            )
            self.change_res_thread.start()

            # whenever resolution changes, calibration parameters no longer apply
            self.monocal.camera.error = None
            self.monocal.camera.camera_matrix = None
            self.monocal.camera.distortion = None
            self.monocal.camera.grid_count = 0
            self.frame_emitter.undistort = False

            self.cal_output.setText(self.monocal.camera.calibration_summary())
            self.clear_grid_history_btn.click()

        self.resolution_combo = QComboBox()

        self.resolution_combo.addItems(resolutions_text())
        self.resolution_combo.setMaximumSize(100, 50)

        w, h = self.monocal.camera.resolution
        self.resolution_combo.setCurrentText(f"{int(w)} x {int(h)}")
        self.resolution_combo.currentTextChanged.connect(change_resolution)


if __name__ == "__main__":
    App = QApplication(sys.argv)

    from queue import Queue

    config_dialogs = []

    from src.calibration.corner_tracker import CornerTracker
    from src.calibration.monocalibrator import MonoCalibrator
    from src.cameras.camera import Camera
    from src.cameras.synchronizer import Synchronizer

    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
    )

    repo = Path(__file__).parent.parent.parent
    config_path = Path(repo, "default_session")
    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_stream_tools()
    session.load_monocalibrators()

    # trackr = CornerTracker(charuco)
    test_port = 0
    # cam = Camera(test_port)
    # stream = VideoStream(cam)
    # streams_dict = {test_port: stream}  # synchronizer expects this format
    # syncr = Synchronizer(streams_dict, fps_target=6)
    # monocal = MonoCalibrator(cam, syncr, trackr)

    logging.info("Creating Camera Config Dialog")
    cam_dialog = CameraConfigDialog(session, test_port)

    logging.info("About to show camera config dialog")
    cam_dialog.show()

    sys.exit(App.exec())
