# Experiment with creating toolbars and menus per:
# https://www.pythonguis.com/tutorials/pyqt6-actions-toolbars-menus/

# %%

import sys
from pathlib import Path

import cv2

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QMainWindow,
    QLabel, QLineEdit, QCheckBox, QScrollArea, QToolBar,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStatusBar)
from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My Awesome App")
        self.setMinimumSize(300,400)

        label = QLabel("Hello!")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(label)

        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)

        button_action = QAction(QIcon(r"src\gui\icons\rotate_right.png"), "Your Button", self)
        button_action.setStatusTip("This is your button")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action.setCheckable(True)
        toolbar.addAction(button_action)

        toolbar.addSeparator()

        button_action2 = QAction(QIcon(r"src\gui\icons\fmc_logo.png"), "Your &button2", self)
        button_action2.setStatusTip("This is your button2")
        button_action2.triggered.connect(self.onMyToolBarButtonClick)
        button_action2.setCheckable(True)
        toolbar.addAction(button_action2)

        toolbar.addWidget(QLabel("Hello  "))
        toolbar.addWidget(QCheckBox())

        toolbar.addSeparator()

        self.setStatusBar(QStatusBar(self))  # weird that we don't create it separately


        menu =  self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action2)
        file_menu.addSeparator()
        file_menu.addAction(button_action)

        submenu = file_menu.addMenu("SubMenu Stuff")
        quit_action = QAction("Close App", self)
        quit_action.triggered.connect(self.close_app)
        submenu.addAction(quit_action)

    def onMyToolBarButtonClick(self, s):
        print("click", s)

            
    def close_app(self):
        self.close()

################### SHOWCASE ################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()
# %%
