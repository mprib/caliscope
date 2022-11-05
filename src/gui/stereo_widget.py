import logging
logging.basicConfig(filename="stereocalibration.log", 
                    filemode = "w", 
                    level=logging.INFO)

import sys
from pathlib import Path
import time

import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap, QFont

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton,
                            QSlider, QComboBox, QDialog, QSizePolicy, QLCDNumber,
                            QToolBar, QLabel, QLineEdit, QCheckBox, QScrollArea, QFileDialog,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QRadioButton)
# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.calibration.stereocalibrator import StereoCalibrator
from src.calibration.synchronizer import Synchronizer
from src.session import Session

class StereoDisplay(QDialog):
    def __init__(self, stereo_cal):
        super(StereoDisplay, self).__init__()

        self.stereo_cal = stereo_cal

        App = QApplication.instance()

        DISPLAY_WIDTH = App.primaryScreen().size().width()
        DISPLAY_HEIGHT = App.primaryScreen().size().height()

        self.setWindowTitle("StereoCamera Calibration")
        self.stacked_frame_width = 500
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)
        self.frame_emitter = StereoFrameEmitter(self.stereo_cal)
        self.frame_emitter.start()
        
        self.setContentsMargins(0,0,0,0)

        self.build_frame_display()

        self.HBL = QHBoxLayout()
        self.setLayout(self.HBL)
        ################### stereo camera scroll area ####################  
        self.scroll = QScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scroll.setWidget(self.frame_display)

        self.HBL.addWidget(self.scroll)

    def build_frame_display(self):
        # return a QLabel that is linked to the constantly changing image
        # IMPORTANT: frame_emitter thread must continue to exist after running
        # this method. Cannot be confined to namespace of the method

        self.frame_display = QLabel()
        self.frame_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        def ImageUpdateSlot(QPixmap):
            self.frame_display.setPixmap(QPixmap)
            self.frame_display.setFixedWidth(QPixmap.width())
            self.frame_display.setFixedHeight(QPixmap.height())

        self.frame_emitter.ImageBroadcast.connect(ImageUpdateSlot)

        def stop_video_slot(bool):
            self.frame_display.setText("Beginning Calibration")
        self.frame_emitter.CalibrationText.connect(stop_video_slot)
class StereoFrameEmitter(QThread):
    # establish signals from the frame that will be displayed in real time 
    # within the GUI
    ImageBroadcast = pyqtSignal(QPixmap)
    CalibrationText = pyqtSignal(object)
    
    def __init__(self, stereo_cal):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(StereoFrameEmitter,self).__init__()
        self.min_sleep = .01 # if true fps drops to zero, don't blow up
        self.stereo_cal = stereo_cal
        logging.info("Initializing Stereo Calibration Frame Emitter")
    
    def run(self):
        self.ThreadActive = True
         
        while self.ThreadActive:
            try:    # takes a moment for capture widget to spin up...don't error out

                # Grab a frame from the capture widget and broadcast to displays
                frame = self.stereo_cal.stacked_frames.get()
                if frame.shape == (1,):     # entered calibration; no more frames
                    self.CalibrationText.emit(True)
                    # self.stop()
                image = self.cv2_to_qlabel(frame)
                pixmap = QPixmap.fromImage(image)

                self.ImageBroadcast.emit(pixmap)

            except AttributeError:
                pass

    def stop(self):
        self.ThreadActive = False
        self.quit()

    def cv2_to_qlabel(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        qt_frame = QImage(Image.data, 
                          Image.shape[1], 
                          Image.shape[0], 
                          QImage.Format.Format_RGB888)
        return qt_frame



if __name__ == "__main__":
    App = QApplication(sys.argv)

    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    session.load_cameras()
    session.load_rtds()
    session.adjust_resolutions()
    # session.find_additional_cameras() # looking to add a third
    start_time = time.perf_counter()


    
    syncr = Synchronizer(session, fps_target=6)
    stereo_cal = StereoCalibrator(syncr)
    stereo_disp = StereoDisplay(stereo_cal)
    stereo_disp.show()


    sys.exit(App.exec())