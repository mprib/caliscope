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
                            QToolBar, QLabel, QLineEdit, QCheckBox, QScrollArea, QSpacerItem,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QSpinBox, QFrame, QFileDialog,
                            QGroupBox, QDoubleSpinBox, QTextEdit, QGraphicsTextItem, QTextBrowser)

from PyQt6.QtMultimedia import QMediaPlayer, QMediaCaptureSession, QVideoFrame
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QImage, QPixmap, QFont, QAction
from cv2 import addWeighted
from pyparsing import java_style_comment

# Append main repo to top of path to allow import of backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.real_time_device import RealTimeDevice
from src.cameras.camera import Camera
from src.calibration.charuco import Charuco, ARUCO_DICTIONARIES

class CharucoBuilder(QDialog):
    def __init__(self):
        super(CharucoBuilder, self).__init__()
        self.setMaximumHeight(DISPLAY_HEIGHT)
        self.setMaximumWidth(DISPLAY_WIDTH/4)

        # Build inputs with sensible defaults; must exist before building board
        self.build_column_spinbox()
        self.build_row_spinbox()
        self.build_width_spinbox()
        self.build_length_spinbox()
        self.build_unit_dropdown()
        self.build_invert_checkbox()

        self.build_config_options()

        # Build primary actions
        self.build_print_btn()
        self.build_true_up_group()
        self.build_export()

        # Build display of board
        self.charuco_added = False  # track to handle redrawing of board
        self.build_charuco()
        self.charuco_added = True

        #################### ESTABLISH LARGELY VERTICAL LAYOUT ##############
        VBL = QVBoxLayout()
        self.setLayout(VBL)
        self.setWindowTitle("Charuco Board Builder")

        ################## STEP ONE: Configure Charuco ######################
        # step1.setFrameStyle(QFrame.)
        # step1.setReadOnly(True)
        step1_text ="""
                    <b>Step 1</b>: Configure the parameters of the charuco 
                    board. The default parameters are appropriate for typical 
                    use cases, though you can invert the board to reduce ink 
                    usage. 
                    """
        step1 = QLabel(step1_text)
        step1.setWordWrap(True)
        # Strange wrinkle: charuco display height and width seem flipped
        step1.setMaximumWidth(self.charuco_display.height()) 
        VBL.addWidget(step1)
        VBL.addWidget(self.charuco_config)
        ##### PRINT ##########################################################
        VBL.addWidget(self.charuco_display)
        VBL.addSpacing(20)
        ############### STEP TWO: Print Charuco  #############################
        step2_text ="""
                    <b>Step 2</b>: Save a picture of the above board to your 
                    computer and then print it out. Tape or glue the paper to 
                    something flat and rigid. You don't want to have any 
                    wrinkles or bends in the board. 
                    
                    The flatness of the board is <b>important</b>.
                    """
        step2 = QLabel(step2_text)
        step2.setWordWrap(True)
        # Strange wrinkle: charuco display height and width seem flipped
        step2.setMaximumWidth(self.charuco_display.height()) 
        VBL.addWidget(step2)
        VBL.addWidget(self.png_btn)
        VBL.addSpacing(20)

        ############### STEP THREE: True-Up to actual Charuco Size ###########
        step3_text ="""
                    <b>Step 3</b>: Measure the actual length of the edge of a 
                    square (in mm) as it appears on the printed board, and input 
                    it into the box below. This adjustment helps to account for 
                    differences in border widths that printers impose and will 
                    ensure that the scale of your motion capture matches the 
                    real world.
                    """
        step3 = QLabel(step3_text)
        step3.setWordWrap(True)
        # Strange wrinkle: charuco display height and width seem flipped
        step3.setMaximumWidth(self.charuco_display.height()) 
        VBL.addWidget(step3)
        VBL.addWidget(self.true_up_group)
        VBL.addSpacing(20)

        ########## STEP FOUR: Export Trued-Up Board to Calibration Folder #####
        step4_text ="""
                    <b>Step 4</b>: After truing up the board, export the board
                    parameters for use in a calibration. NOTE: Save 
                    configurations as <code>.json</code> files.
                    """
        step4 = QLabel(step4_text)
        step4.setWordWrap(True)
        # Strange wrinkle: charuco display height and width seem flipped
        step4.setMaximumWidth(self.charuco_display.height()) 
        VBL.addWidget(step4)
        VBL.addWidget(self.export)

    def build_config_options(self):
        #####################  HORIZONTAL CONFIG BOX  ########################
        config_options = QHBoxLayout()
        config_options.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.charuco_config = QGroupBox("&Configure Charuco Board")
        self.charuco_config.setLayout(config_options)
        self.charuco_config.setCheckable(True)
        self.charuco_config.setChecked(False)    # sensible defaults....


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

        ####################################### UPDATE CHARUCO #################
        self.build_charuco_update_btn()
        config_options.addWidget(self.charuco_build_btn)



    def build_print_btn(self):
        self.png_btn = QPushButton("&Save png") 

        def save_png():
            save_file_tuple = QFileDialog.getSaveFileName(self, "Save As", "charuco.png", "PNG (*.png)")
            print(save_file_tuple)
            save_file_name = str(Path(save_file_tuple[0]))
            if len(save_file_name)>1:
                print(f"Saving board to {save_file_name}")
                self.charuco.save_image(save_file_name)


        self.png_btn.clicked.connect(save_png)

    def build_true_up_group(self):
        self.true_up_group = QGroupBox("True-Up Printed Square Edge")
        self.true_up_group.setLayout(QHBoxLayout())
        self.true_up_group.layout().addWidget(QLabel("Actual Length (mm):"))
        
        self.printed_edge_length = QDoubleSpinBox()
        self.printed_edge_length.setMaximumWidth(100)
        def update_charuco():
            self.charuco.square_size_overide = self.printed_edge_length.value()
            print("Updated Square Size Overide")
        self.printed_edge_length.valueChanged.connect(update_charuco)
        self.true_up_group.layout().addWidget(self.printed_edge_length)

    def build_export(self):
        self.export = QPushButton("&Export")
        
        def export_json():
            save_file_tuple = QFileDialog.getSaveFileName(self, "Save As", "charuco.json", "JSON (*.json)")
            print(save_file_tuple)
            save_file_name = str(Path(save_file_tuple[0]))
            if len(save_file_name)>1:
                print(f"Saving board to {save_file_name}")
                self.charuco.export_as_json(save_file_name)
        self.export.clicked.connect(export_json)

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
        self.invert_checkbox = QCheckBox("&Invert")
        self.invert_checkbox.setChecked(False)

    def build_charuco_update_btn(self):
        self.charuco_build_btn = QPushButton("&Update")
        # self.charuco_build_btn.setText("Create Charuco")
        self.charuco_build_btn.setMaximumSize(50,30)
        self.charuco_build_btn.clicked.connect(self.build_charuco)



    def build_charuco(self):
        ####################### PNG DISPLAY     ###########################
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
        self.charuco = Charuco(columns,
                          rows,
                          board_height,
                          board_width,
                          units = units,
                          dictionary = dictionary_str,
                          aruco_scale = aruco_scale, 
                          square_size_overide = square_edge_length,
                          inverted = inverted)
        # working_charuco_img = cv2.imencode(".png",charuco.board_img) 

        if not self.charuco_added:
            self.charuco_display = QLabel()
            self.charuco_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        charuco_img = self.convert_cv_qt(self.charuco.board_img)
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


   
# %%
