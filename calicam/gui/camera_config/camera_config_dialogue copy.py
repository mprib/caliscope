import calicam.logger
logger = calicam.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread
import time

import cv2
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
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
from calicam.gui.camera_config.frame_emitter import FrameEmitter
from calicam.calibration.monocalibrator import MonoCalibrator
from calicam.cameras.camera import Camera
from calicam.cameras.live_stream import LiveStream 
from calicam.session import Session


class CameraConfigDialog(QDialog):

    def __init__(self, session, port):
        super(CameraConfigDialog, self).__init__()

        # set up variables for ease of reference
        self.session = session
        self.monocal = session.monocalibrators[port]
        self.port = port
        self.stream = self.monocal.stream
        self.camera = self.stream.camera

        App = QApplication.instance()
        self.setWindowTitle("Camera Configuration and Calibration")
        self.setContentsMargins(0, 0, 0, 0)

        ################### BUILD SUB WIDGETS #############################
        
        ###################################################################
        
        self.setLayout(QHBoxLayout())
        ##########################################################
        ###################### CALIBRATION  ################################
        self.build_calibrate_grp()
        self.layout().addWidget(self.calibrate_grp)


        self.frame_controls_layout = QVBoxLayout(self)
        self.layout().addLayout(self.frame_controls_layout)
        self.basic_frame_control = FrameControlWidget(self.session, self.port)
        self.advanced_controls = AdvancedControls(self.session, self.port)
       
        
        self.frame_controls_layout.addWidget(self.basic_frame_control)
        self.frame_controls_layout.addWidget(self.advanced_controls)

        self.frame_controls_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.frame_controls_layout.setContentsMargins(0, 0, 0, 0)

        #################      VIDEO AT TOP     ##########################
        # self.basic_frame_controls.addWidget(self.frame_display)

        ############################  ADD HBOX OF CONFIG ######################

        #######################     FPS   + Grid Count #########################
        # controls = QHBoxLayout()
        # self.build_fps_grp()
        # self.build_grid_group()
        # controls.addWidget(self.fps_grp)
        # controls.addWidget(self.grid_grp)
        # self.frame_controls_layout.addLayout(controls)


    ####################### SUB_WIDGET CONSTRUCTION ###############################

    def save_camera(self):
        self.session.save_camera(self.port)

    def build_calibrate_grp(self):
        logger.debug("Building Calibrate Group")
        self.calibrate_grp = QGroupBox("Calibrate")
        # Generally Horizontal Configuration
        vbox = QVBoxLayout()
        self.calibrate_grp.setLayout(vbox)

        # Collect Calibration Corners
        button_vbox = QVBoxLayout()
        button_vbox.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        collect_crnr_btn = QPushButton("Capture Grids")
        collect_crnr_btn.setMaximumWidth(100)
        button_vbox.addWidget(collect_crnr_btn)

        def capture():
            """change to turn on/off"""
            if not self.monocal.capture_corners:
                self.monocal.capture_corners = True
                collect_crnr_btn.setText("Stop Capture")
                self.calibrate_btn.setEnabled(False)
            else:
                self.monocal.capture_corners = False
                collect_crnr_btn.setText("Capture Grids")
                if self.monocal.grid_count > 1:
                    self.calibrate_btn.setEnabled(True)
                    self.clear_grid_history_btn.setEnabled(True)

        collect_crnr_btn.clicked.connect(capture)

        # Calibrate Button
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setMaximumWidth(100)
        button_vbox.addWidget(self.calibrate_btn)

        def calibrate():
            if len(self.monocal.all_ids) > 0:
                self.cal_output.setText("Calibration can take a moment...")
                self.calibrate_btn.setEnabled(False)
                self.clear_grid_history_btn.setEnabled(True)

                def wrker():
                    self.monocal.calibrate()
                    self.cal_output.setText(self.monocal.camera.calibration_summary())
                    self.save_cal_btn.setEnabled(True)
                    self.undistort_btn.setEnabled(True)

                self.calib_thread = Thread(target=wrker, args=(), daemon=True)
                self.calib_thread.start()
            else:
                self.cal_output.setText("Need to Collect Grids")

        self.calibrate_btn.clicked.connect(calibrate)

        # Clear calibration history
        self.clear_grid_history_btn = QPushButton("Clear History")
        self.clear_grid_history_btn.setMaximumWidth(100)
        self.clear_grid_history_btn.setEnabled(False)
        button_vbox.addWidget(self.clear_grid_history_btn)

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
        if self.monocal.camera.distortion is None:
            self.undistort_btn.setEnabled(False)
        else:
            self.undistort_btn.setEnabled(True)

        self.undistort_btn.setMaximumWidth(100)
        button_vbox.addWidget(self.undistort_btn)

        def undistort():
            def undistort_worker():
                # create thread for timer to play out
                self.frame_emitter.undistort = True
                for i in range(5,0,-1):
                    self.undistort_btn.setText(f"Reverting in {i}")
                    time.sleep(1)
                    
                self.frame_emitter.undistort = False
                self.undistort_btn.setText("Undistort")
            undistort_thread = Thread(target=undistort_worker,args=(),daemon=True)
            undistort_thread.start()

        self.undistort_btn.clicked.connect(undistort)

        def on_save_click():
            self.session.save_camera(self.port)

        # Save Calibration
        self.save_cal_btn = QPushButton("Save Calibration")
        # self.save_cal_btn.setEnabled(False)
        self.save_cal_btn.setMaximumWidth(100)
        self.save_cal_btn.clicked.connect(on_save_click)
        button_vbox.addWidget(self.save_cal_btn)

        # include calibration grid in horizontal box
        vbox.addLayout(button_vbox)

        self.cal_output = QLabel()
        self.cal_output.setWordWrap(True)
        # self.cal_output.setMaximumWidth(int(self.pixmap_edge / 3))
        self.cal_output.setText(self.monocal.camera.calibration_summary())
        vbox.addWidget(self.cal_output)
        # calib_output.setMaximumWidth()



        # self.frame_emitter.GridCountBroadcast.connect(grid_count_update_slot)


class AdvancedControls(QWidget):
    def __init__(self,session:Session, port):
        super(AdvancedControls, self).__init__()
        self.session: Session = session
        self.port = port
        self.monocal: MonoCalibrator = self.session.monocalibrators[port]
        self.stream: LiveStream = self.monocal.stream
        self.camera: Camera = self.stream.camera
        self.setLayout(QHBoxLayout())        

        self.place_widgets()
        self.connect_widgets()
        
    def place_widgets(self):        

        self.fps_grp = QGroupBox("FPS")
        self.layout().addWidget(self.fps_grp)
        self.fps_grp.setLayout(QHBoxLayout())

        logger.debug("Building FPS Control")
        self.fps_grp.layout().addWidget(QLabel("Target:"))

        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(self.stream.fps)
        self.fps_grp.layout().addWidget(self.frame_rate_spin)

        self.fps_display = QLabel()
        # self.fps_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.fps_grp.layout().addWidget(self.fps_display)

        self.grid_grp = QGroupBox("Grid Collection")
        self.layout().addWidget(self.grid_grp)
        self.grid_grp.setLayout(QHBoxLayout())

        self.grid_grp.layout().addWidget(QLabel("Wait Time:"))

        self.wait_time_spin = QDoubleSpinBox()
        self.wait_time_spin.setValue(self.monocal.wait_time)
        self.wait_time_spin.setSingleStep(0.1)

        self.wait_time_spin.valueChanged.connect(self.on_wait_time_spin)
        self.grid_grp.layout().addWidget(self.wait_time_spin)

        logger.debug("Building Grid Count Display")
        self.grid_count_display = QLabel()
        self.grid_grp.layout().addWidget(self.grid_count_display)
        self.grid_count_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        def grid_count_update_slot(grid_count):
            self.grid_count_display.setText(f"Count: {grid_count}")
        # self.frame_emitter.FPSBroadcast.connect(FPSUpdateSlot)

    def on_wait_time_spin(self, wait_time):
        self.monocal.wait_time = wait_time

    def connect_widgets(self):
        self.frame_rate_spin.valueChanged.connect(self.on_frame_rate_spin)
        
    def on_frame_rate_spin(self,fps_rate):
        self.stream.set_fps_target(fps_rate)
        logger.info(f"Changing monocalibrator frame rate for port{self.port}")

    def FPSUpdateSlot(self,fps):
        if self.monocal.camera.capture.isOpened():
            # rounding to nearest integer should be close enough for our purposes
            self.fps_display.setText("Actual: " + str(round(fps, 1)))
        else:
            self.fps_display.setText("reconnecting to camera...")

class FrameControlWidget(QWidget):
    def __init__(self, session: Session, port):
        super(FrameControlWidget, self).__init__()
        self.session:Session = session
        self.monocal:MonoCalibrator  = session.monocalibrators[port]
        self.port = port
        self.camera: Camera = self.monocal.stream.camera

        # need frame emitter to create actual frames    
        DISPLAY_WIDTH = App.primaryScreen().size().width()
        DISPLAY_HEIGHT = App.primaryScreen().size().height()
        self.pixmap_edge = min(DISPLAY_WIDTH / 3, DISPLAY_HEIGHT / 3)
        self.frame_emitter = FrameEmitter(self.monocal, self.pixmap_edge)
        self.frame_emitter.start()
       
       
        self.place_widgets()
        self.connect_widgets()
         
    def place_widgets(self):

        self.setLayout(QVBoxLayout())
        ###### FRAME ####################################
        # return a QLabel that is linked to the constantly changing image
        self.frame_display = QLabel()
        self.layout().addWidget(self.frame_display)

        # self.build_ignore_checkbox()

        self.rotation_resolution_hbox = QHBoxLayout()
        self.layout().addLayout(self.rotation_resolution_hbox)
        ###################### Rotation Buttons  ###################################      
        self.cw_rotation_btn = QPushButton(
            QIcon("calicam/gui/icons/rotate-camera-right.svg"), ""
        )
        self.cw_rotation_btn.setMaximumSize(35,35)
        self.ccw_rotation_btn = QPushButton(
            QIcon("calicam/gui/icons/rotate-camera-left.svg"), ""
        )

        self.ccw_rotation_btn.setMaximumSize(35,35)
        self.rotation_resolution_hbox.addWidget(self.cw_rotation_btn)
        self.rotation_resolution_hbox.addWidget(self.ccw_rotation_btn)

        ##################### RESOLUTION DROP DOWN ##############################
        self.resolution_combo = QComboBox()

        resolutions_text = []
        for w, h in self.monocal.stream.camera.verified_resolutions:
            resolutions_text.append(f"{int(w)} x {int(h)}")

        self.resolution_combo.addItems(resolutions_text)
        self.resolution_combo.setMaximumSize(100, 35)

        w, h = self.monocal.camera.resolution
        self.resolution_combo.setCurrentText(f"{int(w)} x {int(h)}")
    
        self.rotation_resolution_hbox.addWidget(self.resolution_combo)    

        ######################## EXPOSURE BOX #############################
        
        self.exposure_hbox = QHBoxLayout()
        self.layout().addLayout(self.exposure_hbox)
    
        # construct a horizontal widget with label: slider: value display
        self.exposure_label = QLabel("Exposure")
        self.exposure_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-10, 0)
        self.exp_slider.setSliderPosition(int(self.monocal.camera.exposure))
        self.exp_slider.setPageStep(1)
        self.exp_slider.setSingleStep(1)
        self.exp_slider.setMaximumWidth(200)
        self.exposure_number = QLabel()
        self.exposure_number.setText(str(int(self.monocal.camera.exposure)))


        self.exposure_hbox.addWidget(self.exposure_label)
        self.exposure_hbox.addWidget(self.exp_slider)
        self.exposure_hbox.addWidget(self.exposure_number)

        self.exposure_hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
        self.ignore_box = QCheckBox("Ignore", self)
        self.layout().addWidget(self.ignore_box)

    def connect_widgets(self):
        def ImageUpdateSlot(QPixmap):
            self.frame_display.setPixmap(QPixmap)

        self.frame_emitter.ImageBroadcast.connect(ImageUpdateSlot)

        # Counter Clockwise rotation called because the display image is flipped
        self.cw_rotation_btn.clicked.connect(self.monocal.camera.rotate_CCW)
        self.cw_rotation_btn.clicked.connect(self.save_camera)

        # Clockwise rotation called because the display image is flipped
        self.ccw_rotation_btn.clicked.connect(self.monocal.camera.rotate_CW)
        self.ccw_rotation_btn.clicked.connect(self.save_camera)
        
        # resolution combo box
        self.resolution_combo.currentTextChanged.connect(self.change_resolution)

        # exposure slider 
        self.exp_slider.valueChanged.connect(self.update_exposure)
        self.ignore_box.stateChanged.connect(self.ignore_cam)

    def ignore_cam(self,signal):
        if signal == 0:  # not checked
            logger.info(f"Don't ignore camera at port {self.port}")
            self.camera.ignore = False
        else:  # value of checkState() might be 2?
            logger.info(f"Ignore camera at port {self.port}")
            self.camera.ignore = True
        self.save_camera()

    def update_exposure(self, exp):
        self.monocal.camera.exposure = exp
        self.exposure_number.setText(str(exp))
        self.save_camera()

    def change_resolution(self, res_text):
        # call the cam_cap widget to change the resolution, but do it in a
        # thread so that it doesn't halt your progress

        w, h = res_text.split("x")
        w, h = int(w), int(h)
        new_res = (w, h)
        # self.cam_cap.change_resolution(new_res)
        logger.info(
            f"Attempting to change resolution of camera at port {self.port}"
        )

        def change_res_worker(new_res):
            self.monocal.stream.change_resolution(new_res)
            
            # clear out now irrelevant params
            self.camera.camera_matrix=None
            self.camera.distortion=None
            self.camera.error=None
            self.camera.grid_count=None
            self.save_camera()


        self.change_res_thread = Thread(
            target=change_res_worker,
            args=(new_res,),
            daemon=True,
        )
        self.change_res_thread.start()

    def save_camera(self):
        self.session.save_camera(self.port)


if __name__ == "__main__":
    App = QApplication(sys.argv)
    from calicam import __root__
    config_path = Path(__root__, "sessions", "laptop")

    print(config_path)
    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    # session.adjust_resolutions()
    session.load_monocalibrators()

    test_port = 0

    logger.info("Creating Camera Config Dialog")
    cam_dialog = CameraConfigDialog(session, test_port)
    cam_dialog.show()
    
    # frame_control = FrameControlWidget(session, test_port)
    # frame_control.show()

    # adv_control = AdvancedControls(session, test_port)
    # adv_control.show()

    logger.info("About to show camera config dialog")

    sys.exit(App.exec())
