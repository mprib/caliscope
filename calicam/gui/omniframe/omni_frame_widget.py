
import calicam.logger
logger = calicam.logger.get(__name__)

import sys
from pathlib import Path
from threading import Thread
import time

import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QThread
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
from calicam.session import Session
from calicam.gui.omniframe.omni_frame_builder import OmniFrameBuilder
from calicam import __root__


class OmniFrameWidget(QWidget):
    
    def __init__(self,session:Session):

        super(OmniFrameWidget, self).__init__()
        self.session = session
        self.synchronizer = self.session.get_synchronizer()

        self.frame_builder = OmniFrameBuilder(self.synchronizer)
        self.frame_emitter = OmniFrameEmitter(self.frame_builder)

        self.layout_widgets()
        self.connect_widgets()        

    def layout_widgets(self):
        
        self.setLayout(QHBoxLayout())
        omni_frame_display = QLabel()
        self.layout().addWidget(omni_frame_display)

    def connect_widgets(self):

        def ImageUpdateSlot(QPixmap):
            self.omni_frame_display.setPixmap(QPixmap)

        self.frame_emitter.ImageBroadcast.connect(ImageUpdateSlot)
        




class OmniFrameEmitter(QThread):
    ImageBroadcast = pyqtSignal(QPixmap)
    
    def __init__(self, omniframe_builder:OmniFrameBuilder):
        
        super(OmniFrameEmitter,self).__init__()
        self.omniframe_builder = omniframe_builder
        
        
    def run(self):
        
        while True:
            omni_frame = self.omniframe_builder.get_omni_frame()
            image = cv2_to_qlabel(omni_frame)
            
            self.ImageBroadcast.emit(image)
    
    
    
def cv2_to_qlabel(frame):
    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FlippedImage = cv2.flip(Image, 1)

    qt_frame = QImage(
        FlippedImage.data,
        FlippedImage.shape[1],
        FlippedImage.shape[0],
        QImage.Format.Format_RGB888,
    )
    return qt_frame

    
if __name__ == "__main__":
    
    App = QApplication(sys.argv)

    config_path = Path(__root__, "tests", "please work")

    session = Session(config_path)
    session.load_cameras()
    session.load_streams()
    session.adjust_resolutions()


    omni_dialog = OmniFrameWidget(session)
    omni_dialog.show()

    sys.exit(App.exec())