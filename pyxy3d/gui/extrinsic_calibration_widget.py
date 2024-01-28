
import caliscope.logger
logger = caliscope.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread, Event
from time import sleep, perf_counter
from enum import Enum

import cv2
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (
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
from caliscope.session.session import LiveSession
from caliscope.gui.frame_builders.paired_frame_builder import PairedFrameBuilder
from caliscope.cameras.synchronizer import Synchronizer
from caliscope import __root__
from caliscope.gui.navigation_bars import NavigationBarNext

# the boards needed before a pair could be used to bridge pairs without common corners
MIN_THRESHOLD_FOR_EARLY_CALIBRATE = 5


class PossibleActions(Enum):
    CollectData = "Collect Data"
    Terminate = "Terminate"
    Calibrate = "Calibrate"

class ExtrinsicCalibrationWidget(QWidget):
    calibration_complete = Signal()
    calibration_initiated = Signal()
    terminate = Signal()
     
    def __init__(self,session:LiveSession):

        super(ExtrinsicCalibrationWidget, self).__init__()
        self.session = session
        self.synchronizer:Synchronizer = self.session.synchronizer

        logger.info(f"about to check if synchronizer has a sync packet")
        while not hasattr(self.session.synchronizer, "current_sync_packet"):
            logger.info("waiting for synchronizer to create first sync packet")
            sleep(.5)

        self.paired_frame_builder = PairedFrameBuilder(self.synchronizer, board_count_target=30)
        self.paired_frame_emitter = PairedFrameEmitter(self.paired_frame_builder)
        self.paired_frame_emitter.start()

        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(self.synchronizer.fps_target)
        self.board_count_spin = QSpinBox()
        self.board_count_spin.setValue(self.paired_frame_builder.board_count_target)
        
        self.stereo_frame_display = QLabel()
        # self.navigation_bar = NavigationBarNext() 
        self.possible_action = PossibleActions.CollectData
        self.calibrate_collect_btn = QPushButton(self.possible_action.value)
        
        self.place_widgets()
        self.connect_widgets()        

        self.update_btn_eligibility()
    
    def shutdown_threads(self):
        """
        There may be some lingering threads running when the extrinsic calibrator loses focus
        This may be causing python to overload and pyqt to segfault during the calibration process
        if I've moved from the extrinsic calibration widget to a different one...
        """
        logger.info("Unsubscribe paired frame builder from sync notice")
        self.paired_frame_builder.unsubscribe_from_synchronizer()
        logger.info("signal paired frame emitter to stop collecting frames")
        self.paired_frame_emitter.keep_collecting.clear()

    def update_btn_eligibility(self):
        if self.session.is_extrinsic_calibration_eligible():
            self.calibrate_collect_btn.setEnabled(True)
        else:
            self.calibrate_collect_btn.setEnabled(False)
    

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
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignHCenter)
       
        self.layout().addWidget(self.calibrate_collect_btn)



    def connect_widgets(self):
        
        self.calibrate_collect_btn.clicked.connect(self.on_calibrate_collect_click)
        self.paired_frame_emitter.ImageBroadcast.connect(self.ImageUpdateSlot)
        self.paired_frame_emitter.possible_to_initialize_array.connect(self.enable_calibration)
        self.frame_rate_spin.valueChanged.connect(self.synchronizer.set_stream_fps)
        self.board_count_spin.valueChanged.connect(self.update_board_count_target)
        self.paired_frame_emitter.calibration_data_collected.connect(self.initiate_calibration)
    
    def update_board_count_target(self, target):
        self.paired_frame_builder.board_count_target = target
        
    def on_calibrate_collect_click(self):
        if self.possible_action == PossibleActions.CollectData:
            logger.info("Begin collecting calibration data")
            # by default, data saved to session folder
            self.paired_frame_builder.store_points.set()
            extrinsic_calibration_path = Path(self.session.path, "calibration", "extrinsic")
            self.session.start_recording(extrinsic_calibration_path,store_point_history=True)
            self.possible_action = PossibleActions.Terminate
            self.calibrate_collect_btn.setText(self.possible_action.value)
            self.calibrate_collect_btn.setEnabled(True)
            self.navigation_bar.back_btn.setEnabled(False)

        elif self.possible_action == PossibleActions.Terminate:
            logger.info("Terminating current data collection")
            self.terminate.emit()
            self.possible_action = PossibleActions.CollectData
            self.calibrate_collect_btn.setText(self.possible_action.value)

        elif self.possible_action == PossibleActions.Calibrate:
            logger.info("Prematurely end data collection to initiate calibration")
            self.paired_frame_builder.store_points.clear()
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
    ImageBroadcast = Signal(QImage)
    calibration_data_collected = Signal() 
    possible_to_initialize_array = Signal()
    
    def __init__(self, paired_frame_builder:PairedFrameBuilder):
        
        super(PairedFrameEmitter,self).__init__()
        self.paired_frame_builder = paired_frame_builder
        logger.info("Initiated frame emitter")        
        self.keep_collecting = Event() 
        self.collection_complete = False

    def wait_to_next_frame(self):
        """
        based on the next milestone time, return the time needed to sleep so that
        a frame read immediately after would occur when needed
        """
        time = perf_counter()
        fractional_time = time % 1
        all_wait_times = self.paired_frame_builder.milestones - fractional_time
        future_wait_times = all_wait_times[all_wait_times > 0]

        if len(future_wait_times) == 0:
            wait =  1 - fractional_time
        else:
            wait =  future_wait_times[0]
        
        sleep(wait)
        
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
            
            if not possible_to_initialize:
                # check to see if it is now
                if self.paired_frame_builder.possible_to_initialize_array(MIN_THRESHOLD_FOR_EARLY_CALIBRATE):
                    logger.info("Signaling that it is possible to initialize array based on collected data.")
                    possible_to_initialize = True
                    self.possible_to_initialize_array.emit()
                      
            stereo_frame = self.paired_frame_builder.get_stereo_frame()
            self.wait_to_next_frame()
            
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
        from caliscope.configurator import Configurator
        from caliscope.trackers.charuco_tracker import CharucoTracker

        App = QApplication(sys.argv)

        session_path = Path(__root__, "dev","sample_sessions", "257")
        configurator = Configurator(session_path)

        session = LiveSession(configurator)
        # session.load_cameras()
        tracker = CharucoTracker(session.charuco)
        session.load_stream_tools(tracker=tracker)


        stereo_dialog = ExtrinsicCalibrationWidget(session)
        stereo_dialog.show()

        sys.exit(App.exec())
