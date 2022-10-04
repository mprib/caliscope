# Built following the tutorials that begin here: 
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/


# though generally frowned upon, these import all statements save a lot of
# typing and the names in PyQt6 don't class with other names
import sys

from PyQt6.QtGui import *
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                            QWidget, QVBoxLayout, QLineEdit, QMenu)
from PyQt6.QtCore import QSize, Qt
from matplotlib import container


# subclass QMainWindow into our window 
class  MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.label = QLabel("Click in this window")
        self.setCentralWidget(self.label)

    # def contextMenuEvent(self, e):
    #     context = QMenu(self)
    #     context.addAction(QAction("test 1", self))
    #     context.addAction(QAction("test 2", self))
    #     context.addAction(QAction("test 3", self))
    #     context.exec(e.globalPos())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            # handle the left-button press in here
            self.label.setText("mousePressEvent LEFT")

        elif e.button() == Qt.MouseButton.MiddleButton:
            # handle the middle-button press in here.
            self.label.setText("mousePressEvent MIDDLE")

        elif e.button() == Qt.MouseButton.RightButton:
            # handle the right-button press in here.
            self.label.setText("mousePressEvent RIGHT")

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.label.setText("mouseReleaseEvent LEFT")

        elif e.button() == Qt.MouseButton.MiddleButton:
            self.label.setText("mouseReleaseEvent MIDDLE")

        elif e.button() == Qt.MouseButton.RightButton:
            self.label.setText("mouseReleaseEvent RIGHT")

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.label.setText("mouseDoubleClickEvent LEFT")

        elif e.button() == Qt.MouseButton.MiddleButton:
            self.label.setText("mouseDoubleClickEvent MIDDLE")

        elif e.button() == Qt.MouseButton.RightButton:
            self.label.setText("mouseDoubleClickEvent RIGHT")

app = QApplication(sys.argv)    # sys.argv allows passing in args from command line
 
window = MainWindow()   # must appear after the application is initalized
window.show()

app.exec() # 