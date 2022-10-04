from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtWidgets import QApplication, QWidget

import sys
from pathlib import Path

class Window(QWidget):
    def __init__(self):

        super().__init__()

app = QApplication(sys.argv)
engine = QQmlApplicationEngine()
window = Window()
engine.rootContext().setContextProperty('window', window)
engine.load("src/gui/video_streams.qml")

sys.exit(app.exec())