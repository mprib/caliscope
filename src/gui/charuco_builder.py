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
from pyparsing import java_style_comment

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

        # Build inputs with sensible defaults; must exist before building board
        self.build_column_spinbox()
        self.build_row_spinbox()
        self.build_width_spinbox()
        self.build_length_spinbox()
        self.build_unit_dropdown()
        self.build_invert_checkbox()

        # Build display of board
        self.charuco_added = False  # track to handle redrawing of board
        self.build_charuco()
        self.charuco_added = True

        # Build primary actions
        self.build_print_btn()
        self.build_true_up_group()
        self.build_export()



        #################### ESTABLISH LARGELY VERTICAL LAYOUT ##############
        VBL = QVBoxLayout()
        self.setLayout(VBL)
        self.setWindowTitle("Charuco Board Builder")



        ################### HORIZONTAL MAIN INTERACTIONS #####################
        ##### PRINT ##########################################################

        top_actions = QHBoxLayout()
        top_actions.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        top_actions.addWidget(self.print_btn)
        top_actions.addWidget(self.true_up_group)
        top_actions.addWidget(self.export)

        VBL.addLayout(top_actions)

        ####################   DISPLAY CHARUCO  #############################
        VBL.addWidget(self.charuco_display)
        #####################  HORIZONTAL CONFIG BOX  ########################
        config_options = QHBoxLayout()
        config_options.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        charuco_config = QGroupBox("Configure Charuco Board")
        charuco_config.setLayout(config_options)
        charuco_config.setCheckable(True)
        charuco_config.setChecked(False)    # sensible defaults....

        VBL.addWidget(charuco_config)

        ### SHAPE GROUP    ################################################
        shape_grp = QGroupBox("row x col")
        shape_grp.setLayout(QHBoxLayout())
        shape_grp.layout().setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        
        # These are reversed from 'row x col', but this is how it works out
        shape_grp.layout().addWidget(self.row_spin)
        shape_grp.layout().addWidget(self.column_spin)
        
        config_options.addWidget(shape_grp)

        #################### SIZE GROUP #######################################
        size_grp = QGroupBox("Target Board Size")
        size_grp.setLayout(QHBoxLayout())
        size_grp.layout().setAlignment(Qt.AlignmentFlag.AlignHCenter)
 

        size_grp.layout().addWidget(self.width_spin)
        size_grp.layout().addWidget(self.length_spin)
        size_grp.layout().addWidget(self.units)

        config_options.addWidget(size_grp)

        ############################# INVERT ####################################
        self.build_invert_checkbox()
        config_options.addWidget(self.invert_checkbox)  

        ####################################### BUILD CHARUCO #################
        self.build_charuco_build_btn()
        config_options.addWidget(self.charuco_build_btn)

        ################################################## ACTUAL EDGE #####
        # HBL.addWidget(self.get_square_edge_length_input())


    def build_print_btn(self):
        self.print_btn = QPushButton("Save png") 
        # self.print_btn.setMaximumSize(100, 50)


    def build_true_up_group(self):
        self.true_up_group = QGroupBox("True-Up Printed Square Edge")
        self.true_up_group.setLayout(QHBoxLayout())
        self.true_up_group.layout().addWidget(QLabel("Actual Length (mm):"))
        
        self.printed_edge_length = QDoubleSpinBox()
        self.true_up_group.layout().addWidget(self.printed_edge_length)

    def build_export(self):
        self.export = QPushButton("Export")



    def build_column_spinbox(self):
        self.column_spin = QSpinBox()
        self.column_spin.setValue(5)
        self.column_spin.setMaximumWidth(50)
        

    def build_row_spinbox(self):
        self.row_spin = QSpinBox()
        self.row_spin.setValue(7)
        self.row_spin.setMaximumWidth(50)
             
    def build_width_spinbox(self):
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setValue(8.5)
        self.width_spin.setMaximumWidth(50)
        

    def build_length_spinbox(self):
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setValue(11)
        self.length_spin.setMaximumWidth(50)

    def build_unit_dropdown(self):
        self.units = QComboBox()
        self.units.addItems(["mm", "inch"])
        self.units.setCurrentText("inch")
        self.units.setMaximumWidth(100)

    def build_invert_checkbox(self):
        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.setChecked(False)

    def build_charuco_build_btn(self):
        self.charuco_build_btn = QPushButton("Update")
        # self.charuco_build_btn.setText("Create Charuco")
        self.charuco_build_btn.setMaximumSize(50,30)
        self.charuco_build_btn.clicked.connect(self.build_charuco)



    def build_charuco(self):
        ####################### PNG DISPLAY     ###########################
        print("Building Charuco")
        columns = self.column_spin.value()
        rows = self.row_spin.value()
        board_height = self.length_spin.value()
        board_width = self.width_spin.value()
        aruco_scale = 0.75 
        units = self.units.currentText()
        square_edge_length = None
        inverted = self.invert_checkbox.isChecked()
        dictionary_str = "DICT_4X4_1000"

        # print("Calling Charuco Function")
        print(f"Inversion value is {inverted}")
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
        # print("Done with Charuco Function")

        if not self.charuco_added:
            self.charuco_display = QLabel()
            self.charuco_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        charuco_img = self.convert_cv_qt(charuco.board_img)
        self.charuco_display.setPixmap(charuco_img)

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

            p = charuco_QImage.scaled(self.charuco_display.width(),
                                      self.charuco_display.height(),
                                    #   1,
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


   