import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QCheckBox,
    QWidget,
    QPushButton,
    QSlider,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, Slot, Signal
from pyxy3d.gui.prerecorded_intrinsic_calibration.camera_display_widget import (
    CameraDataDisplayWidget,
)

from pyxy3d.controller import Controller
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)


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


class IntrinsicCalibrationWidget(QWidget):
    def __init__(self, controller: Controller, port: int, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.port = port

        self.total_frames = self.controller.get_intrinsic_stream_frame_count(self.port)
        self.frame_image = QLabel(self)
        self.frame_index_label = QLabel(self)
        self.play_button = QPushButton("Play", self)
        self.slider = CustomSlider()
        self.add_grid_btn = QPushButton("Add Grid")
        self.calibrate_btn = QPushButton("Calibrate")
        self.camera_data_display = CameraDataDisplayWidget(self.port, self.controller)
        self.clear_calibration_data_btn = QPushButton("Clear Data")
        self.toggle_distortion = QCheckBox("Apply Distortion")

        self.slider.setMaximum(self.total_frames - 1)
        self.is_playing = False

        self.place_widgets()
        self.connect_widgets()

    def place_widgets(self):
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.camera_data_display)

        self.right_panel = QVBoxLayout()
        self.right_panel.addWidget(self.frame_image, alignment=Qt.AlignmentFlag.AlignCenter)
        self.right_panel.addWidget(self.play_button)
        self.right_panel.addWidget(self.slider)
        self.right_panel.addWidget(self.add_grid_btn)
        self.right_panel.addWidget(self.calibrate_btn)
        self.right_panel.addWidget(self.clear_calibration_data_btn)
        self.right_panel.addWidget(self.frame_index_label)
        self.right_panel.addWidget(self.toggle_distortion)
        self.layout.addLayout(self.right_panel)

    def connect_widgets(self):
        self.play_button.clicked.connect(self.play_video)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.arrowKeyPressed.connect(self.slider_moved)
        self.add_grid_btn.clicked.connect(self.add_grid)
        self.calibrate_btn.clicked.connect(self.calibrate)
        self.clear_calibration_data_btn.clicked.connect(self.clear_calibration_data)
        self.toggle_distortion.stateChanged.connect(self.toggle_distortion_changed)
        
        # self.controller.connect_frame_emitter(self.port, self.update_image,self.update_index)
        self.controller.ImageUpdate.connect(self.update_image)
        self.controller.IndexUpdate.connect(self.update_index)

        # initialize stream to push first frame to widget then hold
        # must be done after signals and slots connected for effect to take hold
        self.controller.play_stream(self.port)

        # self.play_started = True
        self.controller.pause_stream(self.port)
        self.controller.stream_jump_to(self.port, 0)

    def play_video(self):
        # if self.play_started:
        if self.is_playing:
            self.is_playing = False
            self.controller.pause_stream(self.port)
            self.play_button.setText("Play")  # now paused so only option is play
        else:
            self.is_playing = True
            self.controller.unpause_stream(self.port)
            self.play_button.setText("Pause")  # now playing so only option is pause

    def slider_moved(self, position):
        self.controller.stream_jump_to(self.port, position)
        if position == self.total_frames - 1:
            self.controller.pause_stream(self.port)
            self.is_playing = False
            self.play_button.setEnabled(False)
            self.play_button.setText("Play")  # now paused so only option is play
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
                    self.controller.pause_stream(self.port)
                    self.is_playing = False
                    self.play_button.setEnabled(False)
                    self.play_button.setText(
                        "Play"
                    )  # now paused so only option is play

    def update_image(self, port, pixmap):
        if port == self.port:
            self.frame_image.setPixmap(pixmap)

    def add_grid(self):
        self.controller.add_calibration_grid(self.port, self.index)
        self.controller.stream_jump_to(self.port, self.index)

    def calibrate(self):
        self.controller.calibrate_camera(self.port)

    def clear_calibration_data(self):
        self.controller.clear_calibration_data(self.port)
        self.controller.stream_jump_to(self.port, self.index)
       
       
    def toggle_distortion_changed(self, state):
        if state ==2:
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



if __name__ == "__main__":
    from pathlib import Path

    app = QApplication(sys.argv)
    from pyxy3d import __root__
    from pyxy3d.helper import copy_contents

    # Define the input file path here.
    original_workspace_dir = Path(
        __root__, "tests", "sessions", "prerecorded_calibration"
    )
    workspace_dir = Path(
        __root__, "tests", "sessions_copy_delete", "prerecorded_calibration"
    )
    copy_contents(original_workspace_dir, workspace_dir)
    controller = Controller(workspace_dir)
    controller.add_camera_from_source(
        Path(workspace_dir, "calibration", "extrinsic", "port_0.mp4")
    )
    controller.load_intrinsic_streams()

    window = IntrinsicCalibrationWidget(controller=controller, port=0)
    window.resize(800, 600)
    logger.info("About to show window")
    window.show()
    sys.exit(app.exec())
