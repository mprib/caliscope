
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
from calicam.cameras.synchronizer import Synchronizer
from calicam import __root__


class OmniFrameWidget(QWidget):
    
    def __init__(self,session:Session):

        super(OmniFrameWidget, self).__init__()
        self.session = session
        self.synchronizer:Synchronizer = self.session.get_synchronizer()

        self.frame_builder = OmniFrameBuilder(self.synchronizer)
        self.frame_emitter = OmniFrameEmitter(self.frame_builder)
        self.frame_emitter.start()

        self.layout_widgets()
        self.connect_widgets()        

    def layout_widgets(self):
        
        self.setLayout(QVBoxLayout())
        self.omni_frame_display = QLabel()
        self.layout().addWidget(self.omni_frame_display)

    def connect_widgets(self):


        self.frame_emitter.ImageBroadcast.connect(self.ImageUpdateSlot)
        

    def ImageUpdateSlot(self, q_image):
        logger.info("using slot")
        qpixmap = QPixmap.fromImage(q_image)
        logger.info("about to set pixmap")
        self.omni_frame_display.setPixmap(qpixmap)



class OmniFrameEmitter(QThread):
    ImageBroadcast = pyqtSignal(QImage)
    
    def __init__(self, omniframe_builder:OmniFrameBuilder):
        
        super(OmniFrameEmitter,self).__init__()
        self.omniframe_builder = omniframe_builder
        logger.info("Initiated frame emitter")        
        
    def run(self):
        logger.info("ENTERING LOOP")
        while True:
            logger.info("Looping at emitter")
            omni_frame = self.omniframe_builder.get_omni_frame()
            cv2.imshow("omni Frame", omni_frame)

            # logger.info("Successfully pulled omni frame")

            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
            if omni_frame is not None:
                logger.info("convert to qlabel")
                image = cv2_to_qlabel(omni_frame)
                logger.info("convert to pixmap")

                logger.info("broadcast pixmap")
                self.ImageBroadcast.emit(image)
    
    
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

        config_path = Path(__root__, "tests", "please work")

        session = Session(config_path)
        session.load_cameras()
        session.load_streams()
        # session.adjust_resolutions()


        omni_dialog = OmniFrameWidget(session)
        omni_dialog.show()

        sys.exit(App.exec())
