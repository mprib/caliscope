import sys
from PySide6.QtWidgets import QStyle
from PySide6.QtGui import QIcon, QPixmap, QPainter
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QSpinBox,
    QDoubleSpinBox,
    QMainWindow,
    QCheckBox,
    QWidget,
    QPushButton,
    QSlider,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, Slot, Signal, QSize
from caliscope.gui.camera_management.camera_display_widget import (
    CameraDataDisplayWidget,
)
from PySide6.QtSvg import QSvgRenderer
from caliscope.controller import Controller
from caliscope import __root__

import caliscope.logger
logger = caliscope.logger.get(__name__)


def svg_to_pixmap(svg_path: Path, size):
    # Load SVG file
    renderer = QSvgRenderer(str(svg_path))

    # Create an empty QPixmap and fill it with transparent color
    pixmap = QPixmap(size)
    pixmap.fill(Qt.transparent)

    # Render SVG onto QPixmap
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    return pixmap


class CustomSlider(QSlider):
    arrowKeyPressed = Signal(int)  # Custom signal

    def __init__(self):
        super(CustomSlider, self).__init__(Qt.Horizontal)
        self._isDragging = False
        self.valueChanged.connect(self.checkForArrowKey)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._isDragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._isDragging = False
        super().mouseReleaseEvent(event)

    def isDragging(self):
        return self._isDragging

    def isUsingArrowKeys(self):
        return self.hasFocus() and not self._isDragging

    @Slot(int)
    def checkForArrowKey(self, value):
        if self.isUsingArrowKeys():
            self.arrowKeyPressed.emit(value)  # Emit the custom signal

# icons from https://iconoir.com
CAM_ROTATE_RIGHT_PATH = Path(__root__, "caliscope", "gui", "icons", "rotate-camera-right.svg")
CAM_ROTATE_LEFT_PATH = Path(__root__, "caliscope", "gui", "icons", "rotate-camera-left.svg")

class IntrinsicCalibrationWidget(QWidget):
    def __init__(self, controller: Controller, port: int, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.port = port

        self.total_frames = self.controller.get_intrinsic_stream_frame_count(self.port)
        self.frame_image = QLabel(self)
        self.frame_index_label = QLabel(self)
        self.play_button = QPushButton("", self)
        self.play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)

        self.play_button.setIcon(self.play_icon)
        self.slider = CustomSlider()
        self.slider.setMaximum(self.total_frames - 1)

        self.add_grid_btn = QPushButton("Add Grid")
        self.calibrate_btn = QPushButton("Calibrate")
        self.camera_data_display = CameraDataDisplayWidget(self.port, self.controller)
        self.clear_calibration_data_btn = QPushButton("Clear Data")
        self.toggle_distortion = QCheckBox("Apply Distortion")

        self.cw_rotation_btn = QPushButton(QIcon(str(CAM_ROTATE_RIGHT_PATH)), "")
        self.cw_rotation_btn.setMaximumSize(35, 35)
        self.ccw_rotation_btn = QPushButton(QIcon(str(CAM_ROTATE_LEFT_PATH)), "")
        self.ccw_rotation_btn.setMaximumSize(35, 35)

        self.autocalibrate_btn = QPushButton("Autocalibrate")
        self.target_grid_count_spin = QSpinBox(self)
        self.target_grid_count_spin.setMaximumWidth(40)
        self.target_grid_count_spin.setRange(0, 100)
        self.target_grid_count_spin.setValue(40)
        self.target_grid_count_spin.setSingleStep(1)

        self.board_threshold_spin = QDoubleSpinBox(self)
        self.board_threshold_spin.setMaximumWidth(50)
        self.board_threshold_spin.setRange(0, 1)
        self.board_threshold_spin.setValue(.7)
        self.board_threshold_spin.setSingleStep(.1)

        # Create the spinbox
        self.scaling_spin = QSpinBox(self)
        self.scaling_spin.setRange(50, 150)
        self.scaling_spin.setValue(100)
        self.scaling_spin.setSingleStep(5)

        self.is_playing = False

        self.place_widgets()
        self.connect_widgets()

    def place_widgets(self):
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.camera_data_display)
        self.right_panel = QVBoxLayout()
        self.right_panel.addWidget(
            self.frame_image, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.rotate_span = QHBoxLayout()
        self.rotate_span.addWidget(self.cw_rotation_btn)
        self.rotate_span.addWidget(self.ccw_rotation_btn)
        self.right_panel.addLayout(self.rotate_span)

        self.auto_control_span = QHBoxLayout()
        self.auto_control_span.addWidget(QLabel("Target Grid Count:"), alignment=Qt.AlignmentFlag.AlignRight)
        self.auto_control_span.addWidget(self.target_grid_count_spin, alignment=Qt.AlignmentFlag.AlignLeft)
        self.auto_control_span.addWidget(QLabel("Board Threshold:"), alignment=Qt.AlignmentFlag.AlignRight)
        self.auto_control_span.addWidget(self.board_threshold_spin, alignment=Qt.AlignmentFlag.AlignLeft)
        self.auto_control_span.addWidget(self.autocalibrate_btn)
        self.right_panel.addLayout(self.auto_control_span)
        
        self.play_span = QHBoxLayout()
        self.play_span.addWidget(self.play_button)
        self.play_button.setMaximumWidth(35)
        self.play_span.addWidget(self.slider)
        self.right_panel.addLayout(self.play_span)
        
        self.manual_control_span = QHBoxLayout()
        self.manual_control_span.addWidget(self.add_grid_btn)
        self.manual_control_span.addWidget(self.calibrate_btn)
        self.manual_control_span.addWidget(self.clear_calibration_data_btn)
        self.right_panel.addLayout(self.manual_control_span)

        self.distortion_control_span = QHBoxLayout()
        self.distortion_control_span.addWidget(self.toggle_distortion, alignment=Qt.AlignmentFlag.AlignCenter)
        self.distortion_control_span.addWidget(QLabel("Zoom:"), alignment=Qt.AlignmentFlag.AlignRight)
        self.distortion_control_span.addWidget(self.scaling_spin, alignment=Qt.AlignmentFlag.AlignLeft)
        self.distortion_control_span.addWidget(self.frame_index_label)
        self.right_panel.addLayout(self.distortion_control_span)
        self.layout.addLayout(self.right_panel)


    def connect_widgets(self):
        self.play_button.clicked.connect(self.play_video)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.arrowKeyPressed.connect(self.slider_moved)
        self.add_grid_btn.clicked.connect(self.add_grid)
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.clear_calibration_data_btn.clicked.connect(self.clear_calibration_data)
        self.toggle_distortion.stateChanged.connect(self.toggle_distortion_changed)
        self.ccw_rotation_btn.clicked.connect(self.rotate_ccw)
        self.cw_rotation_btn.clicked.connect(self.rotate_cw)
        self.autocalibrate_btn.clicked.connect(self.autocalibrate) 
       
        self.scaling_spin.valueChanged.connect(self.on_scale_change)
        self.controller.intrinsic_stream_manager.frame_emitters[self.port].ImageBroadcast.connect(self.update_image)
        self.controller.intrinsic_stream_manager.frame_emitters[self.port].FrameIndexBroadcast.connect(self.update_index)
        self.controller.enable_inputs.connect(self.update_enable_all_inputs)

        # initialize stream to push first frame to widget then hold
        # must be done after signals and slots connected for effect to take hold
        self.controller.play_intrinsic_stream(self.port)

        # self.play_started = True
        self.controller.pause_intrinsic_stream(self.port)
        self.controller.stream_jump_to(self.port, 0)


    def play_video(self):
        # if self.play_started:
        if self.is_playing:
            self.is_playing = False
            self.controller.pause_intrinsic_stream(self.port)
            self.play_button.setIcon(self.play_icon)
        else:
            self.is_playing = True
            self.controller.unpause_intrinsic_stream(self.port)
            self.play_button.setIcon(self.pause_icon)

    def slider_moved(self, position):
        self.controller.stream_jump_to(self.port, position)
        if position == self.total_frames - 1:
            self.controller.pause_intrinsic_stream(self.port)
            self.is_playing = False
            self.play_button.setEnabled(False)
            self.play_button.setIcon(self.play_icon)
        else:
            if not self.play_button.isEnabled():
                self.play_button.setEnabled(True)

    def update_index(self, port, position):
        """
        only update slider with the position when the stream is making it happen
        track user interact with the widget to assess whether user is currently interacting
        with the slider, at which point don't try to programmatically change the position
        """
        if port == self.port:
            self.index = position
            self.frame_index_label.setText(f"Frame Index: {self.index}")
            if self.slider.isDragging() or self.slider.isUsingArrowKeys():
                pass  # don't change slider position as this would create a feedback loop
            else:
                self.slider.setValue(position)
                if position == self.total_frames - 1:
                    self.controller.pause_intrinsic_stream(self.port)
                    self.is_playing = False
                    self.play_button.setEnabled(False)
                    # now paused so only option is play
                    self.play_button.setIcon(self.play_icon)

    def update_image(self, port, pixmap):
        logger.debug(f"Running `update_image` in playback widget for port {self.port}")
        if port == self.port:
            self.frame_image.setPixmap(pixmap)

    def add_grid(self):
        self.controller.add_calibration_grid(self.port, self.index)
        self.controller.stream_jump_to(self.port, self.index)

    def on_scale_change(self, value):
        """
        Way too much starting to happen here, but there we are...
        """
        new_scale = value/100
        logger.info(f"Changing frame_emitter scale factor to {new_scale}")
        self.controller.scale_intrinsic_stream(self.port, new_scale) 
        self.controller.stream_jump_to(self.port, self.index)

    def calibrate(self):
        
        self.controller.calibrate_camera(self.port)

    def clear_calibration_data(self):
        self.controller.clear_calibration_data(self.port)
        self.controller.stream_jump_to(self.port, self.index)

    def toggle_distortion_changed(self, state):
        if state == 2:
            logger.info("Apply distortion model")
            self.controller.apply_distortion(self.port, True)
            self.controller.stream_jump_to(self.port, self.index)

        else:
            logger.info("Removing distortion")
            self.controller.apply_distortion(self.port, False)
            self.controller.stream_jump_to(self.port, self.index)

    def closeEvent(self, event):
        # self.cap.release()
        super().closeEvent(event)

    def rotate_cw(self):
        self.controller.rotate_camera(self.port, 1)
        self.controller.stream_jump_to(self.port, self.index)

    def rotate_ccw(self):
        self.controller.rotate_camera(self.port, -1)
        self.controller.stream_jump_to(self.port, self.index)


    def update_enable_all_inputs(self, port, enable:bool):
        # Control widget accessibilty from controller signal to all ports
        if port == self.port:
            self.play_button.setEnabled(enable)
            self.slider.setEnabled(enable)
            self.add_grid_btn.setEnabled(enable)
            self.calibrate_btn.setEnabled(enable)
            self.camera_data_display.setEnabled(enable)
            self.clear_calibration_data_btn.setEnabled(enable)
            self.toggle_distortion.setEnabled(enable)
            self.cw_rotation_btn.setEnabled(enable)
            self.ccw_rotation_btn.setEnabled(enable)
            self.autocalibrate_btn.setEnabled(enable)
            self.target_grid_count_spin.setEnabled(enable)
            self.board_threshold_spin.setEnabled(enable)
            self.scaling_spin.setEnabled(enable)
        
    def autocalibrate(self):
        grid_count = self.target_grid_count_spin.value()
        board_threshold = self.board_threshold_spin.value()
        self.update_enable_all_inputs(self.port, False)
        self.controller.autocalibrate(self.port,grid_count, board_threshold)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    from caliscope import __root__
    from caliscope.helper import copy_contents
    from caliscope.trackers.charuco_tracker import CharucoTracker
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
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    charuco_tracker = CharucoTracker(charuco)
    controller.charuco_tracker = charuco_tracker

    controller.load_camera_array()
    controller.load_intrinsic_stream_manager()
    window = IntrinsicCalibrationWidget(controller=controller, port=1)
    window.resize(800, 600)
    logger.info("About to show window")
    window.show()
    sys.exit(app.exec())