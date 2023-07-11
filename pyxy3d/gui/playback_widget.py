
import typing
from PyQt6 import QtCore
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)


from PyQt6.QtGui import QDesktopServices, QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QSizePolicy,
    QWidget,
    QSpinBox,
    QScrollArea,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QLineEdit,
    QListWidget,
    QGroupBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.gui.recording_widget import FramePrepper, FrameDictionaryEmitter
from pyxy3d.gui.vizualize.playback_triangulation_widget import (
    PlaybackTriangulationWidget,
)


class PlaybackWidget(QWidget):
    """
    displays all frames as they are played back by the sychronizer
    provides a way to visualize the quality of the tracking
    
    This is not being used anywhere right now...a small piece of development that
    I'm putting to the side as low priority.    

    """

    def __init__(self,synchronizer:Synchronizer):
        super().__init__()
        
        self.synchronizer = synchronizer
        self.display = QLabel()

        # create tools to build and emit the displayed frame
        self.frame_builder = FramePrepper(self.synchronizer)
        self.frame_emitter = FrameDictionaryEmitter(self.frame_builder)
        self.frame_emitter.start()
       
        
        self.place_widgets()
        self.connect_widgets()
        
        
    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        
        
    def connect_widgets(self):
        
        self.frame_emitter.ThumbnailImagesBroadcast.connect(self.ImageUpdateSlot)
       
    def ImageUpdateSlot(self, q_image):
        self.display.resize(self.display.sizeHint())
        qpixmap = QPixmap.fromImage(q_image)
        self.display.setPixmap(qpixmap)