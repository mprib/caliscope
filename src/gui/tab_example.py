# Built following the tutorials that begin here: 
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QWidget,
)

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.gui.charuco_builder import CharucoBuilder
from src.session import Session
# from PyQt6.layout_colorwidget import Color


class MainWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        app = QApplication.instance()
        screen = app.primaryScreen()
        DISPLAY_WIDTH = screen.size().width()
        DISPLAY_HEIGHT = screen.size().height()         
        self.setMinimumSize(DISPLAY_HEIGHT/3, DISPLAY_WIDTH/3)

        self.setWindowTitle("FreeMocap")
        self.setWindowIcon(QIcon("src/gui/icons/fmc_logo.ico"))

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setMovable(True)

        # tab_names = ["Charuco", 
        #             "Single Camera",
        #             "StereoCalibration",
        #             "Motion Capture"]

        # for name in tab_names:
            # tabs.addTab(MainTab(), name)

        tabs.addTab(CharucoBuilder(self.session), "Charuco Builder")

        self.setCentralWidget(tabs)

class MainTab(QWidget):

    def __init__(self):
        super(MainTab, self).__init__()
        # self.setAutoFillBackground(True)
        # palette = self.palette()
        # palette.setColor(QPalette.ColorRole.Window, QColor(color))
        # self.setPalette(palette)


if __name__ == "__main__":
    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    
    app = QApplication(sys.argv)
    
    window = MainWindow(session)
    window.show()
    
    app.exec()