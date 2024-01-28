import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from caliscope.calibration.charuco import Charuco
from caliscope.controller import Controller
import caliscope.logger

logger = caliscope.logger.get(__name__)
class CharucoWidget(QWidget):
    def __init__(self, controller:Controller):
        super().__init__()

        logger.info("Charuco Wizard initializing")
        self.controller = controller 
        self.params = self.controller.get_charuco_params()

        # add group to do initial configuration of the charuco board
        self.charuco_config = CharucoConfigGroup(self.controller)
        self.charuco_config.row_spin.valueChanged.connect(self.build_charuco)
        self.charuco_config.column_spin.valueChanged.connect(self.build_charuco)
        self.charuco_config.width_spin.valueChanged.connect(self.build_charuco)
        self.charuco_config.length_spin.valueChanged.connect(self.build_charuco)
        self.charuco_config.units.currentIndexChanged.connect(self.build_charuco)
        self.charuco_config.invert_checkbox.stateChanged.connect(self.build_charuco)

        # Build primary actions
        self.build_save_png_group()
        self.build_true_up_group()

        # Build display of board
        self.charuco_added = False  # track to handle redrawing of board
        self.build_charuco()
        self.charuco_added = True

        #################### ESTABLISH LARGELY VERTICAL LAYOUT ##############
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Charuco Board Builder")

        self.layout().addWidget(self.charuco_config)
        self.layout().setAlignment(self.charuco_config, Qt.AlignmentFlag.AlignHCenter)
        self.layout().addWidget(QLabel("<i>Top left corner is point (0,0,0) when setting capture volume origin</i>"))
        self.layout().addWidget(self.charuco_display,2)
        self.layout().addSpacing(10)
        self.layout().addLayout(self.save_png_hbox)
        self.layout().addSpacing(10)
        self.layout().addLayout(self.true_up_hbox)
        self.layout().addWidget(QLabel("<i>Printed square size will set the scale of the capture volume</i>"))


    def build_save_png_group(self):
        # basic png save button
        self.png_btn = QPushButton("Save &png")
        self.png_btn.setMaximumSize(100, 30)

        def save_png():
            save_file_tuple = QFileDialog.getSaveFileName(
                self,
                "Save As",
                str(Path(self.controller.workspace, "charuco.png")),
                "PNG (*.png)",
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
                self,
                "Save As",
                str(Path(self.controller.workspace, "charuco_mirror.png")),
                "PNG (*.png)",
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
        self.true_up_hbox = QHBoxLayout()
        self.true_up_hbox.addWidget(QLabel("Actual Printed Square Edge Length (cm):"))

        self.printed_edge_length = QDoubleSpinBox()
        self.printed_edge_length.setSingleStep(0.01)
        self.printed_edge_length.setMaximumWidth(100)
        # self.set_true_edge_length()
        overide = self.controller.config.dict["charuco"]["square_size_overide_cm"]
        self.printed_edge_length.setValue(overide)

        def update_charuco():
            self.charuco.square_size_overide_cm = round(
                self.printed_edge_length.value(), 2
            )

            logger.info(
                f"Updated Square Size Overide to {self.printed_edge_length.value}"
            )
            self.controller.charuco = self.charuco
            self.controller.config.save_charuco(self.charuco)

        self.printed_edge_length.valueChanged.connect(update_charuco)

        self.true_up_hbox.layout().addWidget(self.printed_edge_length)

    def build_charuco(self):
        columns = self.charuco_config.column_spin.value()
        rows = self.charuco_config.row_spin.value()
        board_height = self.charuco_config.length_spin.value()
        board_width = self.charuco_config.width_spin.value()
        aruco_scale = 0.75
        units = self.charuco_config.units.currentText()
        square_edge_length = self.printed_edge_length.value()
        # a
        inverted = self.charuco_config.invert_checkbox.isChecked()
        dictionary_str = self.params["dictionary"]

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

        # interesting problem comes up when scaling this... I want to switch between scaling the width and height
        # based on how these two things relate....
        if board_height > board_width:
            charuco_height = int(self.height() / 2)
            charuco_width = int(charuco_height * (board_width / board_height))
        else:
            charuco_width = int(self.width() / 2)
            charuco_height = int(charuco_width * (board_height / board_width))

        logger.info("Building charuco thumbnail...")
        charuco_img = self.charuco.board_pixmap(charuco_width, charuco_height)
        self.charuco_display.setPixmap(charuco_img)

        self.controller.update_charuco(self.charuco)


class CharucoConfigGroup(QWidget):
    def __init__(self, controller:Controller):
        super().__init__()
        self.controller = controller
        self.params = self.controller.config.dict["charuco"]

        self.column_spin = QSpinBox()
        self.column_spin.setMinimum(3)
        self.column_spin.setValue(self.params["columns"])
        # self.column_spin.setMaximumWidth(50)

        self.row_spin = QSpinBox()
        self.row_spin.setMinimum(4)
        self.row_spin.setValue(self.params["rows"])
        # self.row_spin.setMaximumWidth(50)

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

if __name__ == "__main__":



    app = QApplication(sys.argv)
    from caliscope import __root__
    from caliscope.calibration.charuco import Charuco

    # Define the input file path here.
    original_workspace_dir = Path(
        __root__, "tests", "sessions", "prerecorded_calibration"
    )
    # workspace_dir = Path(
    #     __root__, "tests", "sessions_copy_delete", "prerecorded_calibration"
    # )

    # copy_contents(original_workspace_dir, workspace_dir)
    workspace_dir = Path(r"C:\Users\Mac Prible\OneDrive\caliscope\prerecorded_workflow")
    controller = Controller(workspace_dir)
    charuco_page = CharucoWidget(controller)

    controller.load_intrinsic_stream_manager()
    window = CharucoWidget(controller=controller)
    window.resize(800, 600)
    logger.info("About to show window")
    window.show()
    sys.exit(app.exec())