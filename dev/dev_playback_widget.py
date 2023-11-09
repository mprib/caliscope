import sys
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel)
from PySide6.QtCore import (Qt, QTimer)
from PySide6.QtGui import (QPixmap, QImage)


from pyxy3d.controller import Controller
import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

class VideoPlayer(QWidget):
    def __init__(self, controller:Controller, port:int,parent=None):
        super().__init__(parent)
        self.controller = controller
        self.port = port 
        self.total_frames = self.controller.get_intrinsic_stream_frame_count(self.port)
        self.frame_image = QLabel(self)

        self.play_button = QPushButton("Play", self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMaximum(self.total_frames)
        self.play_started = False
        self.is_playing = False

        self.place_widgets()
        self.connect_widgets()
     
    def place_widgets(self):
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.frame_image, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)
    
    def connect_widgets(self):
        self.play_button.clicked.connect(self.play_video)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.controller.connect_frame_emitter(self.port, self.display_image)
        
    def play_video(self):
        if self.play_started:
            if self.is_playing:
                self.is_playing = False    
                self.controller.pause_stream(self.port)
                self.play_button.setText("Play") # now paused so only option is play
            else:
                self.is_playing = True
                self.controller.unpause_stream(self.port)
                self.play_button.setText("Pause")  # now playing so only option is pause
        else:
            self.play_started = True
            self.is_playing = True
            logger.info(f"Initiate stream playback at port {self.port}")
            self.controller.play_stream(self.port)
            self.play_button.setText("Pause")  # now playing so only option is pause
            
    def next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)
            self.slider.setValue(self.slider.value() + 1)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.slider.setValue(0)

    def slider_moved(self, position):
        self.controller.stream_jump_to(self.port, position)

    def display_image(self, pixmap):
        self.frame_image.setPixmap(pixmap)

    def closeEvent(self, event):
        # self.cap.release()
        super().closeEvent(event)

class VideoWindow(QMainWindow):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Player")
        self.player = VideoPlayer(video_path, self)
        self.setCentralWidget(self.player)

if __name__ == "__main__":
    # from pyxy3d.configurator import Configurator
    from pathlib import Path
    app = QApplication(sys.argv)
    from pyxy3d import __root__
    from pyxy3d.helper import copy_contents
    # Define the input file path here.
    original_workspace_dir = Path(__root__,"tests","sessions", "prerecorded_calibration")
    workspace_dir = Path(__root__,"tests","sessions_copy_delete", "prerecorded_calibration")
    copy_contents(original_workspace_dir,workspace_dir)
    controller = Controller(workspace_dir)
    controller.add_camera_from_source(Path(workspace_dir,"calibration", "extrinsic", "port_0.mp4"))
    controller.load_intrinsic_streams()
    
    window = VideoPlayer(controller=controller,port=0)
    window.resize(800, 600)
    logger.info("About to show window")
    window.show()
    sys.exit(app.exec())
