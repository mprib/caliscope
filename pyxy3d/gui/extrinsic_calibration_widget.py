
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread, Event
import time
from enum import Enum

import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QSizePolicy,
    QWidget,
    QSpinBox,
    QScrollArea,
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
from pyxy3d.session.session import Session
from pyxy3d.gui.frame_builders.paired_frame_builder import PairedFrameBuilder
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.gui.navigation_bars import NavigationBarNext

# the boards needed before a pair could be used to bridge pairs without common corners
MIN_THRESHOLD_FOR_EARLY_CALIBRATE = 5


class PossibleActions(Enum):
    CollectData = "Collect Data"
    Terminate = "Terminate"
    Calibrate = "Calibrate"

class ExtrinsicCalibrationWidget(QWidget):
    calibration_complete = pyqtSignal()
    calibration_initiated = pyqtSignal()
    terminate = pyqtSignal()
     
    def __init__(self,session:Session):

        super(ExtrinsicCalibrationWidget, self).__init__()
        self.session = session
        self.synchronizer:Synchronizer = self.session.synchronizer

        self.create_stereoframe_tools()

        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(self.synchronizer.fps_target)
        self.board_count_spin = QSpinBox()
        self.board_count_spin.setValue(self.frame_builder.board_count_target)
        
        self.stereo_frame_display = QLabel()
        # self.navigation_bar = NavigationBarNext() 
        self.possible_action = PossibleActions.CollectData
        self.calibrate_collect_btn = QPushButton(self.possible_action.value)
        
        self.place_widgets()
        self.connect_widgets()        

    def create_stereoframe_tools(self):

        self.frame_builder = PairedFrameBuilder(self.synchronizer, board_count_target=30)
        self.frame_emitter = PairedFrameEmitter(self.frame_builder)
        self.frame_emitter.start()

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        
        self.settings_group = QGroupBox("Settings")
        self.settings_group.setLayout(QHBoxLayout())
        self.settings_group.layout().addWidget(QLabel("Frame Rate:"))
        self.settings_group.layout().addWidget(self.frame_rate_spin)       
        self.settings_group.layout().addWidget(QLabel("Target Board Count:"))
        self.settings_group.layout().addWidget(self.board_count_spin)       

        self.layout().addWidget(self.settings_group)
        # self.layout().addWidget(self.calibrate_collect_btn)

        # scroll bar appears to not be working....
        self.scroll_area = QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # self.scroll_area.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll_area)

        self.scroll_area.setWidget(self.stereo_frame_display)
       
        self.layout().addWidget(self.calibrate_collect_btn)



    def connect_widgets(self):
        
        self.calibrate_collect_btn.clicked.connect(self.on_calibrate_collect_click)
        self.frame_emitter.ImageBroadcast.connect(self.ImageUpdateSlot)
        self.frame_emitter.possible_to_initialize_array.connect(self.enable_calibration)
        self.frame_rate_spin.valueChanged.connect(self.synchronizer.set_stream_fps)
        self.board_count_spin.valueChanged.connect(self.update_board_count_target)
        self.frame_emitter.calibration_data_collected.connect(self.initiate_calibration)
    
    def update_board_count_target(self, target):
        self.frame_builder.board_count_target = target
        
    def on_calibrate_collect_click(self):
        if self.possible_action == PossibleActions.CollectData:
            logger.info("Begin collecting calibration data")
            # by default, data saved to session folder
            self.frame_builder.store_points.set()
            extrinsic_calibration_path = Path(self.session.path, "calibration", "extrinsic")
            self.session.start_recording(extrinsic_calibration_path,store_point_history=True)
            self.possible_action = PossibleActions.Terminate
            self.calibrate_collect_btn.setText(self.possible_action.value)
            self.calibrate_collect_btn.setEnabled(True)
            self.navigation_bar.back_btn.setEnabled(False)

        elif self.possible_action == PossibleActions.Terminate:
            logger.info("Terminating current data collection")
            self.terminate.emit()
            # self.session.stop_recording()
            # self.frame_builder.reset()
            self.possible_action = PossibleActions.CollectData
            self.calibrate_collect_btn.setText(self.possible_action.value)

        elif self.possible_action == PossibleActions.Calibrate:
            logger.info("Prematurely end data collection to initiate calibration")
            self.frame_builder.store_points.clear()
            self.initiate_calibration()
            


    def enable_calibration(self):
        self.possible_action = PossibleActions.Calibrate
        self.calibrate_collect_btn.setText(self.possible_action.value)
        self.calibrate_collect_btn.setEnabled(True)
        
        
    def ImageUpdateSlot(self, q_image):
        self.stereo_frame_display.resize(self.stereo_frame_display.sizeHint())

        qpixmap = QPixmap.fromImage(q_image)
        self.stereo_frame_display.setPixmap(qpixmap)
        


    def initiate_calibration(self):
        def worker():
            self.calibration_initiated.emit()
            logger.info("Beginning wind-down process prior to calibration")
            self.calibrate_collect_btn.setText("---calibrating---")
            self.calibrate_collect_btn.setEnabled(False)
            # self.frame_emitter.stop()
            # self.stereo_frame_display.hide()
            logger.info("Stop recording video")
            self.session.stop_recording()
            logger.info("Begin calibration")
            logger.info("Pause synchronizer")
            self.session.pause_synchronizer()
            self.session.estimate_extrinsics()

            self.possible_action = PossibleActions.CollectData
            self.calibrate_collect_btn.setText(self.possible_action.value)
            self.calibrate_collect_btn.setEnabled(True)
            self.calibration_complete.emit()
            logger.info("Calibration Complete signal sent...")
            
        self.init_calibration_thread = Thread(target=worker,args=(), daemon=True)
        self.init_calibration_thread.start()

class PairedFrameEmitter(QThread):
    ImageBroadcast = pyqtSignal(QImage)
    calibration_data_collected = pyqtSignal() 
    possible_to_initialize_array = pyqtSignal()
    
    def __init__(self, paired_frame_builder:PairedFrameBuilder):
        
        super(PairedFrameEmitter,self).__init__()
        self.paired_frame_builder = paired_frame_builder
        logger.info("Initiated frame emitter")        
        self.keep_collecting = Event() 
        self.collection_complete = False
        
    def run(self):

        self.keep_collecting.set()
        self.collection_complete = False

        possible_to_initialize = False
        
        while self.keep_collecting.is_set():
            
            # that that it is important to make sure that this signal is sent only once
            # to avoid multiple calibration attempts 
            if len(self.paired_frame_builder.stereo_list) == 0 and not self.collection_complete:
                logger.info("Signalling that calibration data is fully collected.")
                self.collection_complete = True
                self.calibration_data_collected.emit()
        
                # break
            
            if not possible_to_initialize:
                # check to see if it is now
                if self.paired_frame_builder.possible_to_initialize_array(MIN_THRESHOLD_FOR_EARLY_CALIBRATE):
                    logger.info("Signaling that it is possible to initialize array based on collected data.")
                    possible_to_initialize = True
                    self.possible_to_initialize_array.emit()
                      
            stereo_frame = self.paired_frame_builder.get_stereo_frame()

            if stereo_frame is not None:
                image = cv2_to_qlabel(stereo_frame)
                self.ImageBroadcast.emit(image)

        logger.info("Stereoframe emitter run thread ended...") 
            
    # def stop(self):
        # self.keep_collecting.clear() 


        
        
        
def cv2_to_qlabel(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    qt_frame = QImage(
        image.data,
        image.shape[1],
        image.shape[0],
        QImage.Format.Format_RGB888,
    )
    return qt_frame

    
if __name__ == "__main__":
        from pyxy3d.configurator import Configurator
        from pyxy3d.trackers.charuco_tracker import CharucoTracker

        App = QApplication(sys.argv)

        session_path = Path(__root__, "dev","sample_sessions", "257")
        configurator = Configurator(session_path)

        session = Session(configurator)
        # session.load_cameras()
        tracker = CharucoTracker(session.charuco)
        session.load_stream_tools(tracker=tracker)


        stereo_dialog = ExtrinsicCalibrationWidget(session)
        stereo_dialog.show()

        sys.exit(App.exec())
