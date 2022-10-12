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

        ###################### VERTICAL PARENT LAYOUT  ####################
        VBL = QVBoxLayout()
        self.setLayout(VBL)
        
        ######################  HORIZONTAL BOX  ###########################
        HBL = QHBoxLayout()
        VBL.addLayout(HBL)
        ### COLUMNS #######################################################
        HBL.addWidget(self.get_column_spinbox())
        ############   ROWS   #############################################
        HBL.addWidget(self.get_row_spinbox())
        ###################### DICTIONARY #################################
        # HBL.addWidget(self.get_charuco_dict_dropdown())
        # ##################################  ARUCO_SCALE ###################
        # HBL.addWidget(self.get_aruco_scale_spinbox())
        # ############################################## MEASURED SQUARE EDGE
        # HBL.addWidget(self.get_square_edge_length_input())


        ####################### PNG DISPLAY     ###########################
        columns = 4
        rows = 5
        square_length = 0.0525
        aruco_length = 0.75
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        
        self.board = cv2.aruco.CharucoBoard_create(
            columns,
            rows,
            square_length,
            aruco_length,
            dictionary)

        charuco_img = self.board.draw((int(width_inch*300), int(height_inch*300)))

        board_display = QLabel()
        board_display.setPixmap(QPixmap(charuco_img))
        VBL.addWidget(board_display)
         
        
    def get_column_spinbox(self):
        column_spin = QSpinBox()
        column_spin.setValue(4)
    
        return column_spin

    def get_row_spinbox(self):
        row_spin = QSpinBox()
        row_spin.setValue(5)
        return row_spin

    # def get_charuco_dict_dropdown(self):
    # 
App = QApplication(sys.argv)
charuco_window = CharucoBuilder()

charuco_window.show()

sys.exit(App.exec())


   