import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

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
from pyxy3d.gui.camera_config.frame_emitter import FrameEmitter
from pyxy3d.calibration.monocalibrator import MonoCalibrator
from pyxy3d.cameras.camera import Camera
from pyxy3d.cameras.live_stream import LiveStream 
from pyxy3d.session.session import Session
from pyxy3d.gui.camera_config.camera_summary_widget import SummaryWidget
from pyxy3d import __root__


class CameraConfigDialog(QDialog):
    
    def __init__(self, session, port):
        super(CameraConfigDialog, self).__init__()

        # set up variables for ease of reference
        self.session = session
        self.monocal = session.monocalibrators[port]
        self.port = port
        self.stream = self.monocal.stream
        self.camera = self.stream.camera

        # need frame emitter to create actual frames and track FPS/grid count 
        App = QApplication.instance()
        DISPLAY_WIDTH = App.primaryScreen().size().width()
        DISPLAY_HEIGHT = App.primaryScreen().size().height()

        self.pixmap_edge = min(DISPLAY_WIDTH / 3, DISPLAY_HEIGHT / 3)
        self.frame_emitter = FrameEmitter(self.monocal, self.pixmap_edge)
        self.frame_emitter.start()

        self.setWindowTitle("Camera Configuration and Calibration")
        self.setContentsMargins(0, 0, 0, 0)

        ################### BUILD SUB WIDGETS #############################
        
        ###################################################################
        
        self.setLayout(QHBoxLayout())
        ##########################################################
        ###################### CALIBRATION  ################################
        self.calibrate_grp = CalibrationControls(self.session, self.port, self.frame_emitter)

        # self.build_calibrate_grp()
        self.layout().addWidget(self.calibrate_grp)


        self.frame_controls_layout = QVBoxLayout(self)
        self.layout().addLayout(self.frame_controls_layout)
        self.basic_frame_control = FrameControlWidget(self.session, self.port, self.frame_emitter)
       
        
        self.frame_controls_layout.addWidget(self.basic_frame_control)

        self.frame_controls_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.frame_controls_layout.setContentsMargins(0, 0, 0, 0)

        self.advanced_controls_toggle = QPushButton("Advanced Settings")
        self.advanced_controls_toggle.setCheckable(True)
        self.advanced_controls_toggle.clicked.connect(self.toggle_advanced_controls)

        self.frame_controls_layout.addWidget(self.advanced_controls_toggle)
    
        self.advanced_controls = AdvancedControls(self.session, self.port, self.frame_emitter)
        self.advanced_controls.hide()
        self.frame_controls_layout.addWidget(self.advanced_controls)

        self.connect_widgets()

    def toggle_advanced_controls(self):
        if self.advanced_controls_toggle.isChecked():
            self.advanced_controls.show()
        else:
            self.advanced_controls.hide()
   
    def connect_widgets(self):
        self.basic_frame_control.resolution_combo.currentTextChanged.connect(self.calibrate_grp.clear_camera_calibration)
        self.calibrate_grp.signal_calibration_lock.connect(self.lock_resolution_combo)
    
    def lock_resolution_combo(self, lock:bool):
        # note, the is locking, so the signal meaning gets reversed when setting "enable"
        enabled = not lock
        self.basic_frame_control.resolution_combo.setEnabled(enabled)
        
    def save_camera(self):
        self.session.save_camera(self.port)

    

class CalibrationControls(QGroupBox):
    signal_calibration_lock = pyqtSignal(bool)
    calibration_change = pyqtSignal()

    def __init__(self, session:Session, port, frame_emitter:FrameEmitter):
        super(CalibrationControls,self).__init__("Calibration Summary")

        self.session: Session = session
        self.port = port
        self.monocal: MonoCalibrator = self.session.monocalibrators[port]
        self.stream: LiveStream = self.monocal.stream
        self.camera: Camera = self.stream.camera
        self.frame_emitter = frame_emitter
        self.setLayout(QVBoxLayout())        

        self.place_widgets()
        self.connect_widgets()


    def place_widgets(self):

        self.start_stop_calibration_btn = QPushButton("&Collect Data")
        self.layout().addWidget(self.start_stop_calibration_btn)
        self.undistort_btn = QPushButton("Un&distort")    
        self.layout().addWidget(self.undistort_btn)
         
        if self.camera.matrix is None and self.camera.distortions is None:
            self.undistort_btn.setEnabled(False)
        
        self.camera_summary = SummaryWidget(self.camera)
        self.layout().addWidget(self.camera_summary)

    def connect_widgets(self):
        self.start_stop_calibration_btn.clicked.connect(self.capture_control)
        self.undistort_btn.clicked.connect(self.undistort)
        self.calibration_change.connect(self.update_camera_data)

    def capture_control(self):
        """change to turn on/off"""

        if self.start_stop_calibration_btn.text() == "&Collect Data":
            self.signal_calibration_lock.emit(True)
            self.clear_camera_calibration()
            self.calibration_change.emit()
            self.monocal.capture_corners.set()
            self.undistort_btn.setEnabled(False)
            self.start_stop_calibration_btn.setText("&Calibrate")
        
        elif self.start_stop_calibration_btn.text() == "&Calibrate":
            self.signal_calibration_lock.emit(True)
            if len(self.monocal.all_ids) > 0:
                # stop the colletion of more data
                self.monocal.capture_corners.clear()
                self.calibrate()    
            else:
                # no data collected
                self.monocal.capture_corners.clear()
                # self.camera_summary.place_widgets()
                self.update_camera_data()
                self.start_stop_calibration_btn.setText("&Collect Data")

        elif self.start_stop_calibration_btn.text() == "Re-&Collect":
            self.signal_calibration_lock.emit(True)
            self.clear_camera_calibration()
            self.calibration_change.emit()
            self.monocal.initialize_grid_history()
            self.undistort_btn.setEnabled(False)
            self.monocal.capture_corners.set()
            self.start_stop_calibration_btn.setText("&Calibrate")
    
    def clear_camera_calibration(self):
        self.camera.matrix = None
        self.camera.distortions = None
        self.camera.error = None
        self.camera.grid_count = None
        self.session.config.save_camera(self.camera)
        # self.camera_summary.place_widgets()
        self.update_camera_data()
        self.undistort_btn.setEnabled(False)
       
    def calibrate(self):

            def wrker():
                self.start_stop_calibration_btn.setText("---processing---")
                self.start_stop_calibration_btn.setEnabled(False)

                self.monocal.calibrate()
                self.session.config.save_camera(self.camera)
                # self.camera_summary.place_widgets()
                # self.update_camera_data()
                # signal to camera tabs to check on total session calibration status
                self.calibration_change.emit() 

                self.undistort_btn.setEnabled(True)
                self.start_stop_calibration_btn.setText("Re-&Collect")
                self.start_stop_calibration_btn.setEnabled(True)
                self.signal_calibration_lock.emit(False)

            self.calib_thread = Thread(target=wrker, args=(), daemon=True)
            self.calib_thread.start()

    def undistort(self):
        if self.undistort_btn.text() == "Un&distort":
            self.signal_calibration_lock.emit(True)
            self.start_stop_calibration_btn.setEnabled(False)
            self.frame_emitter.undistort = True
            self.undistort_btn.setText("Revert &Distortion")
                
        elif self.undistort_btn.text() == "Revert &Distortion":
            self.start_stop_calibration_btn.setEnabled(True)
            self.frame_emitter.undistort = False
            self.undistort_btn.setText("Un&distort")
            self.signal_calibration_lock.emit(False)
            
    def update_camera_data(self):
        self.layout().removeWidget(self.camera_summary)
        self.camera_summary.deleteLater()
        self.camera_summary = None
        self.camera_summary = SummaryWidget(self.camera)
        self.layout().addWidget(self.camera_summary)
        
        
class AdvancedControls(QWidget):
    def __init__(self,session:Session, port, frame_emitter:FrameEmitter):
        super(AdvancedControls, self).__init__()
        self.session: Session = session
        self.port = port
        self.monocal: MonoCalibrator = self.session.monocalibrators[port]
        self.stream: LiveStream = self.monocal.stream
        self.camera: Camera = self.stream.camera
        self.frame_emitter = frame_emitter
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

        # logger.debug("Building Grid Count Display")
        # self.grid_count_display = QLabel()
        # self.grid_grp.layout().addWidget(self.grid_count_display)
        # self.grid_count_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # def grid_count_update_slot(grid_count):
        #     self.grid_count_display.setText(f"Count: {grid_count}")
        # self.frame_emitter.FPSBroadcast.connect(FPSUpdateSlot)

    def connect_widgets(self):
        self.frame_rate_spin.valueChanged.connect(self.on_frame_rate_spin)
        
        
        def FPSUpdateSlot(fps):
            if self.monocal.camera.capture.isOpened():
                # rounding to nearest integer should be close enough for our purposes
                self.fps_display.setText("Actual: " + str(round(fps, 1)))
            else:
                self.fps_display.setText("reconnecting to camera...")

        self.frame_emitter.FPSBroadcast.connect(FPSUpdateSlot)
        
    def on_wait_time_spin(self, wait_time):
        self.monocal.wait_time = wait_time

        
    def on_frame_rate_spin(self,fps_rate):
        # self.stream.set_fps_target(fps_rate)
        self.monocal.set_stream_fps(fps_rate)
        logger.info(f"Changing monocalibrator frame rate for port{self.port}")

    def FPSUpdateSlot(self,fps):
        if self.monocal.camera.capture.isOpened():
            # rounding to nearest integer should be close enough for our purposes
            self.fps_display.setText("Actual: " + str(round(fps, 1)))
        else:
            self.fps_display.setText("reconnecting to camera...")

class FrameControlWidget(QWidget):
    def __init__(self, session: Session, port, frame_emitter:FrameEmitter):
        super(FrameControlWidget, self).__init__()
        self.session:Session = session
        self.monocal:MonoCalibrator  = session.monocalibrators[port]
        self.port = port
        self.camera: Camera = self.monocal.stream.camera
        self.frame_emitter = frame_emitter
       
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
        # icons from https://iconoir.com
        self.cw_rotation_btn = QPushButton(
            QIcon(str(Path(__root__, "pyxy3d/gui/icons/rotate-camera-right.svg"))), ""
        )
        self.cw_rotation_btn.setMaximumSize(35,35)
        self.ccw_rotation_btn = QPushButton(
            QIcon(str(Path(__root__, "pyxy3d/gui/icons/rotate-camera-left.svg"))), ""
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

        w, h = self.monocal.camera.size
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

    def save_camera(self):
        # normally wouldn't bother with a one-liner function, but it makes connecting
        # to the signal more straightforward
        self.session.config.save_camera(self.camera)
    
    
    def ignore_cam(self,signal):
        if signal == 0:  # not checked
            logger.info(f"Don't ignore camera at port {self.port}")
            self.camera.ignore = False
        else:  # value of checkState() might be 2?
            logger.info(f"Ignore camera at port {self.port}")
            self.camera.ignore = True
        # self.session.config.save_camera(self.camera)
        self.save_camera()

    def update_exposure(self, exp):
        self.monocal.camera.exposure = exp
        self.exposure_number.setText(str(exp))
        self.save_camera()
        # self.session.config.save_camera(self.camera)

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
            self.camera.matrix=None
            self.camera.distortions=None
            self.camera.error=None
            self.camera.grid_count=None
            # self.session.config.save_camera(self.camera)
            self.save_camera()

        self.change_res_thread = Thread(
            target=change_res_worker,
            args=(new_res,),
            daemon=True,
        )
        self.change_res_thread.start()


if __name__ == "__main__":
    from pyxy3d import __root__
    from pyxy3d.configurator import Configurator
    from pyxy3d.trackers.charuco_tracker import CharucoTracker

    import toml
    from pyxy3d import __app_dir__

    app_settings = toml.load(Path(__app_dir__, "settings.toml"))
    recent_projects:list = app_settings["recent_projects"]

    recent_project_count = len(recent_projects)
    session_path = Path(recent_projects[recent_project_count-1])

    config = Configurator(session_path)
    session = Session(config)

    tracker = CharucoTracker(session.charuco)
    # # session.load_cameras()
    session.load_stream_tools(tracker=tracker)
    test_port = 0

    App = QApplication(sys.argv)

    logger.info("Creating Camera Config Dialog")
    cam_dialog = CameraConfigDialog(session, test_port)
    cam_dialog.show()
    
    logger.info("About to show camera config dialog")

    sys.exit(App.exec())
