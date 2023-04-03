
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread, Event
import time

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
from pyxy3d.session import Session
from pyxy3d.gui.stereoframe.stereo_frame_builder import StereoFrameBuilder
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.gui.qt_logger import QtLogger
from pyxy3d.gui.widgets import NavigationBarBackFinish

class StereoFrameWidget(QWidget):
    calibration_complete = pyqtSignal(bool)

    def __init__(self,session:Session):

        super(StereoFrameWidget, self).__init__()
        self.session = session
        self.synchronizer:Synchronizer = self.session.get_synchronizer()

        self.create_stereoframe_tools()

        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(self.synchronizer.get_fps_target())
        self.board_count_spin = QSpinBox()
        self.board_count_spin.setValue(self.frame_builder.board_count_target)
        
        self.stereo_frame_display = QLabel()
        self.navigation_bar = NavigationBarBackFinish() 
        self.calibrate_collect_btn = self.navigation_bar.calibrate_collect_btn

        
        self.place_widgets()
        self.connect_widgets()        

    def create_stereoframe_tools(self):

        self.frame_builder = StereoFrameBuilder(self.synchronizer, board_count_target=10)
        self.frame_emitter = StereoFrameEmitter(self.frame_builder)
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
       
        self.layout().addWidget(self.navigation_bar)



    def connect_widgets(self):
        
        self.calibrate_collect_btn.clicked.connect(self.on_calibrate_connect_click)
        self.frame_emitter.ImageBroadcast.connect(self.ImageUpdateSlot)
        self.frame_rate_spin.valueChanged.connect(self.synchronizer.set_fps_target)
        self.board_count_spin.valueChanged.connect(self.update_board_count_target)
        self.frame_emitter.calibration_data_collected.connect(self.initiate_calibration)
    
    def update_board_count_target(self, target):
        self.frame_builder.board_count_target = target
        
    def on_calibrate_connect_click(self):
        if self.calibrate_collect_btn.text() == "Collect Data":
            logger.info("Begin collecting calibration data")
            # by default, data saved to session folder
            self.frame_builder.store_points.set()
            self.session.start_recording()
            self.calibrate_collect_btn.setText("Early Terminate")
        elif self.calibrate_collect_btn.text() == "Early Terminate":
            logger.info("Prematurely end data collection")
            self.frame_builder.store_points.clear()
            self.initiate_calibration()
        # elif self.calibrate_collect_btn.text() == "Calibrate": 
            # self.session.pause_synchronizer()
            # self.initiate_calibration()

    def ImageUpdateSlot(self, q_image):
        self.stereo_frame_display.resize(self.stereo_frame_display.sizeHint())

        qpixmap = QPixmap.fromImage(q_image)
        self.stereo_frame_display.setPixmap(qpixmap)
        


    def initiate_calibration(self):
        def worker():
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
            self.calibration_complete.emit(True)
            
            
        self.init_calibration_thread = Thread(target=worker,args=(), daemon=True)
        self.init_calibration_thread.start()

class StereoFrameEmitter(QThread):
    ImageBroadcast = pyqtSignal(QImage)
    calibration_data_collected = pyqtSignal(bool) 

    def __init__(self, stereoframe_builder:StereoFrameBuilder):
        
        super(StereoFrameEmitter,self).__init__()
        self.stereoframe_builder = stereoframe_builder
        logger.info("Initiated frame emitter")        
        self.keep_collecting = Event() 
        self.collection_complete = False
        
    def run(self):

        self.keep_collecting.set()
        self.collection_complete = False

        while self.keep_collecting.is_set():
            
            # that that it is important to make sure that this signal is sent only once
            # to avoid multiple calibration attempts 
            if len(self.stereoframe_builder.stereo_list) == 0 and not self.collection_complete:
                logger.info("Signalling that calibration data is fully collected.")
                self.collection_complete = True
                self.calibration_data_collected.emit(True)
                # break
            
            stereo_frame = self.stereoframe_builder.get_stereo_frame()

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
        App = QApplication(sys.argv)

        config_path = Path(__root__, "tests", "217")

        session = Session(config_path)
        session.load_cameras()
        session.load_streams()
        session.adjust_resolutions()


        stereo_dialog = StereoFrameWidget(session)
        stereo_dialog.show()

        sys.exit(App.exec())
