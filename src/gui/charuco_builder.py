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
                            QVBoxLayout, QHBoxLayout, QGridLayout, QSpinBox, 
                            QGroupBox, QDoubleSpinBox)

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
        charuco_config = QGroupBox("Configure Charuco Board")
        charuco_config.setCheckable(True)
        HBL = QHBoxLayout()
        charuco_config.setLayout(HBL)
        HBL.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        VBL.addWidget(charuco_config)

        ### SHAPE GROUP    ################################################
        shape_grp = QGroupBox("row x col")
        shape_grp.setLayout(QHBoxLayout())
        shape_grp.layout().setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        self.build_column_spinbox()
        self.build_row_spinbox()
        
        shape_grp.layout().addWidget(self.column_spin)
        shape_grp.layout().addWidget(self.row_spin)
        
        HBL.addWidget(shape_grp)

        #################### SIZE GROUP #######################################
        size_grp = QGroupBox("Target Board Size")
        size_grp.setLayout(QHBoxLayout())
        size_grp.layout().setAlignment(Qt.AlignmentFlag.AlignHCenter)
 
        self.build_width_spinbox()
        self.build_length_spinbox()
        self.build_unit_dropdown()

        size_grp.layout().addWidget(self.width_spin)
        size_grp.layout().addWidget(self.length_spin)
        size_grp.layout().addWidget(self.units)

        HBL.addWidget(size_grp)

        #############################   


        ####################   DISPLAY CHARUCO  #############################
        self.build_charuco()
        VBL.addWidget(self.charuco_display)
        ################################################## ACTUAL EDGE #####
        # HBL.addWidget(self.get_square_edge_length_input())
             
    def build_width_spinbox(self):
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setValue(4)
        self.width_spin.setMaximumWidth(50)
        

    def build_length_spinbox(self):
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setValue(5)
        self.length_spin.setMaximumWidth(50)

    def build_unit_dropdown(self):
        self.units = QComboBox()
        self.units.addItems(["mm", "inch"])
        self.units.setMaximumWidth(100)
    
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
        rows = 7
        board_height = 11
        board_width = 8
        aruco_scale = 0.75 
        units = "inches" 
        square_edge_length = 0.0525
        aruco_length = 0.75
        inverted = False
        dictionary_str = "DICT_7X7_1000"
        
        charuco = Charuco(columns,
                          rows,
                          board_height,
                          board_width,
                          units = units,
                          dictionary = dictionary_str,
                          aruco_scale = aruco_scale, 
                          square_size_overide = square_edge_length,
                          inverted = inverted)
        # working_charuco_img = cv2.imencode(".png",charuco.board_img) 
        
        self.charuco_display = QLabel()
        # charuco_img = cv2.imread(charuco.board_img)
        charuco_img = self.convert_cv_qt(charuco.board_img)
        self.charuco_display.setPixmap(charuco_img)
        self.charuco_display.setScaledContents(True)
        self.charuco_display.setMaximumSize(self.width() - 10, self.height() - 20)


    def convert_cv_qt(self, cv_img):
            """Convert from an opencv image to QPixmap"""
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            charuco_QImage = QImage(rgb_image.data, 
                                          w, 
                                          h, 
                                          bytes_per_line, 
                                          QImage.Format.Format_RGB888)

            p = charuco_QImage.scaled(self.width(), 
                                      self.height(),
                                      Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)

            return QPixmap.fromImage(p)
            # return QPixmap.fromImage(charuco_QImage)



if __name__ == "__main__":
    App = QApplication(sys.argv)


    screen = App.primaryScreen()
    DISPLAY_WIDTH = screen.size().width()
    DISPLAY_HEIGHT = screen.size().height()

    charuco_window = CharucoBuilder()

    charuco_window.show()

    sys.exit(App.exec())


   