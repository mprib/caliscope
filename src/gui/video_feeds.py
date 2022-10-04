from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, 
    QLabel, QLineEdit, 
    QVBoxLayout, QHBoxLayout, QGridLayout
)


from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from pathlib import Path
import sys

# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# import src.concurrency_tutorial.video_stream_widget as vWidget

# from qtpy import QT_API

class Window(QWidget):
    def __init__(self):
        super().__init__()
    
        self.setWindowTitle("FreeMoCap")
        self.setWindowIcon(QIcon(r"src\gui\icons\fmc_logo.ico"))
        # self.setGeometry()
        self.create_player()
        
    def create_player(self):
        self.mediaPlayer = QMediaPlayer(None) 

        videowidget = QVideoWidget()

        self.openBtn = QPushButton('Open Video')
        # self.openBtn.setEnabled(False)
        # self. 
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        hbox.addWidget(self.openBtn)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)

        self.setLayout(vbox)

# https://www.youtube.com/watch?v=45sPjuPJ3vs
# at ~ five minutes, thirty seconds



app = QApplication(sys.argv)


window = Window()
window.show()

sys.exit(app.exec())