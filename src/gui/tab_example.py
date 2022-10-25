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

# from PyQt6.layout_colorwidget import Color


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        app = QApplication.instance()
        screen = app.primaryScreen()
        DISPLAY_WIDTH = screen.size().width()
        DISPLAY_HEIGHT = screen.size().height()         
        self.

        self.setWindowTitle("FreeMocap")
        self.setWindowIcon(QIcon("src/gui/icons/fmc_logo.ico"))

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setMovable(True)

        tab_names = ["Charuco", 
                    "Single Camera",
                    "StereoCalibration",
                    "Motion Capture"]
        for name in tab_names:
            tabs.addTab(MainTab(), name)

        self.setCentralWidget(tabs)

class MainTab(QWidget):

    def __init__(self):
        super(MainTab, self).__init__()
        # self.setAutoFillBackground(True)
        # palette = self.palette()
        # palette.setColor(QPalette.ColorRole.Window, QColor(color))
        # self.setPalette(palette)




app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()