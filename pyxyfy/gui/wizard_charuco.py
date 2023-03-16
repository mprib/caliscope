import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from pyxy3d import __app_dir__
from pyxy3d.calibration.charuco import ARUCO_DICTIONARIES, Charuco
from pyxy3d.session import Session
from pyxy3d.gui.qt_logger import QtLogger
from pyxy3d.gui.widgets import NavigationBarNext

class WizardCharuco(QWidget):
    def __init__(self, session):
        super().__init__()

        logger.info("Charuco Wizard initializing")
        self.session = session
        self.params = self.session.config["charuco"]
        
        # add group to do initial configuration of the charuco board
        self.configurator = CharucoConfigurator(self.session)
        self.configurator.row_spin.valueChanged.connect(self.build_charuco)
        self.configurator.column_spin.valueChanged.connect(self.build_charuco)
        self.configurator.width_spin.valueChanged.connect(self.build_charuco)
        self.configurator.length_spin.valueChanged.connect(self.build_charuco)
        self.configurator.units.currentIndexChanged.connect(self.build_charuco)
        self.configurator.invert_checkbox.stateChanged.connect(self.build_charuco)

        # Build primary actions
        self.build_save_png_group()
        self.build_true_up_group()
        # self.build_save_config()
        # Build display of board
        self.charuco_added = False  # track to handle redrawing of board
        self.build_charuco()
        self.charuco_added = True

        #################### ESTABLISH LARGELY VERTICAL LAYOUT ##############
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Charuco Board Builder")

        self.layout().addWidget(self.configurator)
        self.layout().addWidget(self.charuco_display)
        self.layout().addSpacing(20)
        self.layout().addLayout(self.save_png_hbox)
        self.layout().addSpacing(20)

        self.layout().addWidget(self.true_up_group)
        self.layout().addSpacing(20)
        for w in self.children():
            self.layout().setAlignment(w, Qt.AlignmentFlag.AlignHCenter)
        
        # add navigation bar at the end so as to not mess up alignment
        self.navigation_bar = NavigationBarNext()
        self.layout().addWidget(self.navigation_bar)


        
        
    def build_save_png_group(self):
        # basic png save button
        self.png_btn = QPushButton("Save &png")
        self.png_btn.setMaximumSize(100, 30)

        def save_png():
            save_file_tuple = QFileDialog.getSaveFileName(
                self, "Save As", str(Path(self.session.path,"charuco.png")), "PNG (*.png)"
            )
            print(save_file_tuple)
            save_file_name = str(Path(save_file_tuple[0]))
            if len(save_file_name) > 1:
                print(f"Saving board to {save_file_name}")
                self.charuco.save_image(save_file_name)

        self.png_btn.clicked.connect(save_png)

        # additional mirror image option
        self.png_mirror_btn = QPushButton("Save &mirror png")
        self.png_mirror_btn.setMaximumSize(100, 30)

        def save_mirror_png():
            save_file_tuple = QFileDialog.getSaveFileName(
                self, "Save As", str(Path(self.session.path,"charuco_mirror.png")), "PNG (*.png)"
            )
            print(save_file_tuple)
            save_file_name = str(Path(save_file_tuple[0]))
            if len(save_file_name) > 1:
                print(f"Saving board to {save_file_name}")
                self.charuco.save_mirror_image(save_file_name)

        self.png_mirror_btn.clicked.connect(save_mirror_png)

        self.save_png_hbox = QHBoxLayout()
        self.save_png_hbox.addWidget(self.png_btn)
        self.save_png_hbox.addWidget(self.png_mirror_btn)

    def build_true_up_group(self):
        self.true_up_group = QGroupBox("&True-Up Printed Square Edge")
        self.true_up_group.setLayout(QHBoxLayout())
        self.true_up_group.layout().addWidget(QLabel("Actual Length (cm):"))

        self.printed_edge_length = QDoubleSpinBox()
        self.printed_edge_length.setSingleStep(.01)
        self.printed_edge_length.setMaximumWidth(100)
        # self.set_true_edge_length()
        overide = self.session.config["charuco"]["square_size_overide_cm"]
        self.printed_edge_length.setValue(overide)

        def update_charuco():
            self.charuco.square_size_overide_cm = round(self.printed_edge_length.value(),2)

            logger.info(f"Updated Square Size Overide to {self.printed_edge_length.value}")
            self.session.charuco = self.charuco
            self.session.save_charuco()

        self.printed_edge_length.valueChanged.connect(update_charuco)

        self.true_up_group.layout().addWidget(self.printed_edge_length)

    # def build_save_config(self):
    #     self.save_btn = QPushButton("&Save Charuco")
    #     self.save_btn.setMaximumSize(100, 30)

    #     def save_charuco():
    #         self.session.charuco = self.charuco
    #         self.session.save_charuco()
    #     self.save_btn.clicked.connect(save_charuco)


    def build_charuco(self):
        columns = self.configurator.column_spin.value()
        rows = self.configurator.row_spin.value()
        board_height = self.configurator.length_spin.value()
        board_width = self.configurator.width_spin.value()
        aruco_scale = 0.75
        units = self.configurator.units.currentText()
        square_edge_length = self.printed_edge_length.value()

        inverted = self.configurator.invert_checkbox.isChecked()
        dictionary_str = "DICT_4X4_1000"

        self.charuco = Charuco(
            columns,
            rows,
            board_height,
            board_width,
            units=units,
            dictionary=dictionary_str,
            aruco_scale=aruco_scale,
            square_size_overide_cm=square_edge_length,
            inverted=inverted,
        )

        if not self.charuco_added:
            self.charuco_display = QLabel()
            self.charuco_display.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            # self.charuco_display.setMaximumSize(
            #     int(self.height() / 2), int(self.width() / 2)
            # )
        
        
        # interesting problem comes up when scaling this... I want to switch between scaling the width and height
        # based on how these two things relate....
        if board_height>board_width:
            charuco_height = int(self.height() / 2)
            charuco_width = int(charuco_height*(board_width/board_height))
        else:
            charuco_width = int(self.width() / 2)
            charuco_height = int(charuco_width*(board_height/board_width))

        logger.info("Building charuco thumbnail...")
        charuco_img = self.charuco.board_pixmap(charuco_width, charuco_height)
        self.charuco_display.setPixmap(charuco_img)

        self.session.charuco = self.charuco
        self.session.save_charuco()

class CharucoConfigurator(QWidget):
   
    def __init__(self, session): 
        super().__init__()
        self.session = session
        self.params = self.session.config["charuco"]
  
        self.column_spin = QSpinBox()
        self.column_spin.setMinimum(2)
        self.column_spin.setValue(self.params["columns"])
        self.column_spin.setMaximumWidth(50)

        self.row_spin = QSpinBox()
        self.row_spin.setMinimum(2)
        self.row_spin.setValue(self.params["rows"])
        self.row_spin.setMaximumWidth(50)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setMinimum(1)
        self.width_spin.setValue(self.params["board_width"])
        self.width_spin.setMaximumWidth(50)

        self.length_spin = QDoubleSpinBox()
        self.length_spin.setMinimum(1)
        self.length_spin.setValue(self.params["board_height"])
        self.length_spin.setMaximumWidth(50)

        self.units = QComboBox()
        self.units.addItems(["cm", "inch"])
        self.units.setCurrentText(self.params["units"])
        self.units.setMaximumWidth(100)

        self.invert_checkbox = QCheckBox("&Invert")
        self.invert_checkbox.setChecked(self.params["inverted"])

        #####################  HORIZONTAL CONFIG BOX  ########################
        self.config_options = QHBoxLayout()
        self.config_options.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.charuco_config = QGroupBox("&Configure Charuco Board")
        self.setLayout(self.config_options)

        ### SHAPE GROUP    ################################################
        shape_grp = QGroupBox("row x col")
        shape_grp.setLayout(QHBoxLayout())
        shape_grp.layout().setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # These are reversed from 'row x col', but this is how it works out
        shape_grp.layout().addWidget(self.row_spin)
        shape_grp.layout().addWidget(self.column_spin)

        self.config_options.addWidget(shape_grp)

        #################### SIZE GROUP #######################################
        size_grp = QGroupBox("Target Board Size")
        size_grp.setLayout(QHBoxLayout())
        size_grp.layout().setAlignment(Qt.AlignmentFlag.AlignHCenter)

        size_grp.layout().addWidget(self.width_spin)
        size_grp.layout().addWidget(self.length_spin)
        size_grp.layout().addWidget(self.units)

        self.config_options.addWidget(size_grp)

        ############################# INVERT ####################################
        self.config_options.addWidget(self.invert_checkbox)

        ####################################### UPDATE CHARUCO #################

        # self.charuco_build_btn = QPushButton("&Update")
        # self.charuco_build_btn.setMaximumSize(50, 30)
        # self.charuco_build_btn.clicked.connect(self.build_charuco)
        # self.config_options.addWidget(self.charuco_build_btn)
        
    def update_charuco(self):
        
        columns = self.column_spin.value()
        rows = self.row_spin.value()
        board_height = self.length_spin.value()
        board_width = self.width_spin.value()
        aruco_scale = 0.75
        units = self.units.currentText()


if __name__ == "__main__":
    
    from pyxy3d import __root__
    config_path = Path(__root__, "tests", "pyxy3d")

    session = Session(config_path)

    app = QApplication(sys.argv)

    charuco_page = WizardCharuco(session)
    charuco_page.show()

    # configurator = CharucoConfigurator(session)
    # configurator.show()

    sys.exit(app.exec())
