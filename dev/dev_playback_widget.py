import sys
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel)
from PySide6.QtCore import (Qt, QTimer)
from PySide6.QtGui import (QPixmap, QImage)


from pyxy3d.controller import Controller

class VideoPlayer(QWidget):
    def __init__(self, controller:Controller, camera_index:int,parent=None):
        super().__init__(parent)
        self.controller = controller
        self.camera_index = camera_index 

        self.frame_image = QLabel(self)
        self.play_button = QPushButton("Play", self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMaximum(self.total_frames)

        self.is_playing = False

        self.place_widgets()
        self.connect_widgets()

    def place_widgets(self):
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.frame_image)
        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)
    
    def connect_widgets(self):
        self.play_button.clicked.connect(self.play_video)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.timer.timeout.connect(self.next_frame)
        
        
    def play_video(self):
        if self.is_playing:
            self.is_playing = False    
            self.controller.pause_stream(self.camera_index)
            self.play_button.setText("Pause")
        else:
            self.is_playing = True
            self.controller.unpause_stream(self.camera_index)
            self.play_button.setText("Pause")

    def next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)
            self.slider.setValue(self.slider.value() + 1)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.slider.setValue(0)

    def slider_moved(self, position):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)

    def display_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.frame_image.setPixmap(pixmap.scaled(self.frame_image.width(), self.frame_image.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
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

    # Define the input file path here.
    workspace_dir = Path(r"C:\Users\Mac Prible\repos\pyxy3d\tests\sessions\4_cam_recording")
    input_file = Path(r"C:\Users\Mac Prible\repos\pyxy3d\tests\sessions\4_cam_recording\calibration\extrinsic\port_0.mp4")
    
    controller = Controller(workspace_dir)
    

    window = VideoWindow(input_file)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
