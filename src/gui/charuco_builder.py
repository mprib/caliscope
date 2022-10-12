# Want to create a small window that allows for the charuco params to be set
# and a board to be constructed that will be used for the calibration

#%%
from ast import arg
import sys
from pathlib import Path
import time
from threading import Thread

import cv2

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton,
                            QSlider, QComboBox, QDialog, QSizePolicy, QLCDNumber,
                            QToolBar, QLabel, QLineEdit, QCheckBox, QScrollArea,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QSpinBox)

from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap, QFont
from cv2 import addWeighted

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera_capture_widget import CameraCaptureWidget
from src.cameras.camera import Camera
from src.calibration.charuco import Charuco, ARUCO_DICTIONARIES

class CharucoBuilder(QDialog):
    def __init__(self):
        super(CharucoBuilder, self).__init__()
        self.setMaximumHeight(DISPLAY_HEIGHT/3)
        self.setMaximumWidth(DISPLAY_WIDTH/4)
        ###################### VERTICAL PARENT LAYOUT  ####################
        VBL = QVBoxLayout()
        self.setLayout(VBL)
        
        ######################  HORIZONTAL BOX  ###########################
        HBL = QHBoxLayout()
        VBL.addLayout(HBL)
        HBL.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        ### COLUMNS #######################################################
        self.build_column_spinbox()
        HBL.addWidget(self.column_spin)
        ########### ROWS ###############################################
        self.build_row_spinbox()
        HBL.addWidget(self.row_spin)

        ############### WIDTH ##############################################

        ##################### HEIGHT #######################################

        
        ############################# DICTIONARY #######################
        # HBL.addWidget(self.get_charuco_dict_dropdown())
        # #####################################  ARUCO_SCALE ###############
        # HBL.addWidget(self.get_aruco_scale_spinbox())
        ################################################## ACTUAL EDGE #####
        # HBL.addWidget(self.get_square_edge_length_input())

        self.build_charuco()
        VBL.addWidget(self.charuco_display)
         
    def build_column_spinbox(self):
        self.column_spin = QSpinBox()
        self.column_spin.setValue(4)
        self.column_spin.setMaximumWidth(50)
        

    def build_row_spinbox(self):
        self.row_spin = QSpinBox()
        self.row_spin.setValue(5)
        self.row_spin.setMaximumWidth(50)

    def build_charuco(self):
        ####################### PNG DISPLAY     ###########################
        columns = 4
        rows = 5
        board_height = 11
        board_width = 8
        
        square_length = 0.0525
        aruco_length = 0.75
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        
        charuco = Charuco(4,5,11,8.5,aruco_scale = .75, square_size_overide=.0525)
        charuco.save_image('test_charuco.png')
        self.charuco_display = QLabel()
        pixmap = QPixmap('test_charuco.png')
        self.charuco_display.setPixmap(pixmap)
        self.charuco_display.setScaledContents(True)
        self.charuco_display.setMaximumSize(self.width() - 10, self.height() - 20)
    
    
App = QApplication(sys.argv)


screen = App.primaryScreen()
DISPLAY_WIDTH = screen.size().width()
DISPLAY_HEIGHT = screen.size().height()

charuco_window = CharucoBuilder()

charuco_window.show()

sys.exit(App.exec())


   