

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
import math
from pathlib import Path
from threading import Thread, Event
import numpy as np
import time
from queue import Queue

import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QSizePolicy,
    QWidget,
    QSpinBox,
    QScrollArea,
    QComboBox,
    QCheckBox,
    QDialog,
    QGroupBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from pyxy3d.session import Session
from pyxy3d.gui.stereoframe.stereo_frame_builder import StereoFrameBuilder
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.gui.qt_logger import QtLogger
from pyxy3d.gui.widgets import NavigationBarBackFinish
from pyxy3d.recording.video_recorder import VideoRecorder

class RecordingWidget(QWidget):
     
    def __init__(self,session:Session):

        super(RecordingWidget, self).__init__()
        self.session = session
        self.synchronizer:Synchronizer = self.session.get_synchronizer()
        
        # don't let point tracking slow down the frame reading
        self.synchronizer.set_tracking_on_streams(False)

        # create tools to build and emit the displayed frame
        self.frame_builder = RecordingFrameBuilder(self.synchronizer)
        self.frame_emitter = RecordingFrameEmitter(self.frame_builder)
        self.frame_emitter.start()

        self.video_recorder = VideoRecorder(self.synchronizer)

        self.recording_destination = QLabel()
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(self.synchronizer.get_fps_target())

        self.start_stop = QPushButton("Start Recording")
        
        self.dropped_fps_label = QLabel()
                
        self.recording_frame_display = QLabel()
        
        self.place_widgets()
        self.connect_widgets()        


    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.settings_group = QGroupBox("Settings")
        self.settings_group.setLayout(QHBoxLayout())
        self.settings_group.layout().addWidget(QLabel("Frame Rate:"))
        self.settings_group.layout().addWidget(self.frame_rate_spin)       
        self.layout().addWidget(self.settings_group)

        self.record_controls = QGroupBox()
        self.record_controls.setLayout(QHBoxLayout())
        self.record_controls.layout().addWidget(self.start_stop)
        self.record_controls.layout().addWidget(self.recording_destination)

        self.layout().addWidget(self.record_controls)
        self.layout().addWidget(self.dropped_fps_label)

        self.layout().addWidget(self.recording_frame_display)


    def connect_widgets(self):
    
        self.frame_emitter.ImageBroadcast.connect(self.ImageUpdateSlot)
        self.frame_rate_spin.valueChanged.connect(self.synchronizer.set_fps_target)
        self.frame_emitter.dropped_fps.connect(self.update_dropped_fps)
        self.start_stop.clicked.connect(self.toggle_start_stop)

    def toggle_start_stop(self):
        if self.start_stop.text() == "Start Recording":
            self.start_stop.setText("Stop Recording")
            logger.info("Initiate recording")

        elif self.start_stop.text() == "Stop Recording":
            self.start_stop.setText("Start Recording")
            logger.info("Stop recording and initiate final save of file") 
        
    def update_dropped_fps(self, dropped_fps:dict):
        "Unravel dropped fps dictionary to a more readable string"
        text = "Rate of Frame Dropping by Port:    "
        
        
        for port, drop_rate in dropped_fps.items():
            text += f"{port}: {drop_rate:.0%}        "

        self.dropped_fps_label.setText(text)
         
        
    def ImageUpdateSlot(self, q_image):
        self.recording_frame_display.resize(self.recording_frame_display.sizeHint())
        qpixmap = QPixmap.fromImage(q_image)
        self.recording_frame_display.setPixmap(qpixmap)
        
class RecordingFrameBuilder:
    def __init__(self, synchronizer: Synchronizer, single_frame_height=250):
        self.synchronizer = synchronizer 
        self.single_frame_height = single_frame_height

        self.rotation_counts = {}
        self.ports = []        
        for port, stream in self.synchronizer.streams.items():
            # override here while testing this out with pre-recorded video
            self.rotation_counts[port] = stream.camera.rotation_count
            self.ports.append(port)
        self.ports.sort()
        
        # reasonable default for the shape of the all-cameras frame
        # make it as square as you can get it
        camera_count = len(self.ports)
        self.frame_columns = int(math.ceil(camera_count**.5))
        self.frame_rows = int(math.ceil(camera_count/self.frame_columns))

        self.new_sync_packet_notice = Queue()
        self.synchronizer.subscribe_to_notice(self.new_sync_packet_notice)
    
    def unsubscribe_from_synchronizer(self):
        logger.info("Unsubscribe frame builder from synchronizer.")
        self.synchronizer.unsubscribe_to_notice(self.new_sync_packet_notice) 


    def get_frame_or_blank(self, frame_packet):
        """Synchronization issues can lead to some frames being None among
        the synched frames, so plug that with a blank frame"""

        edge = self.single_frame_height

        if frame_packet is None:
            logger.debug("plugging blank frame data")
            frame = np.zeros((edge, edge, 3), dtype=np.uint8)
        else:
            frame = frame_packet.frame

        return frame


    def resize_to_square(self, frame):
        """To make sure that frames align well, scale them all to thumbnails
        squares with black borders."""
        logger.debug("resizing square")

        frame = cv2.flip(frame, 1)

        height = frame.shape[0]
        width = frame.shape[1]

        padded_size = max(height, width)

        height_pad = int((padded_size - height) / 2)
        width_pad = int((padded_size - width) / 2)
        pad_color = [0, 0, 0]

        logger.debug("about to pad border")
        frame = cv2.copyMakeBorder(
            frame,
            height_pad,
            height_pad,
            width_pad,
            width_pad,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        frame = resize(frame, new_height=self.single_frame_height)
        return frame


    def apply_rotation(self, frame, port):
        rotation_count = self.rotation_counts[port]
        if rotation_count == 0:
            pass
        elif rotation_count in [1, -3]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_count in [2, -2]:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_count in [-1, 3]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame


    def get_recording_frame(self):
        """
        This glues together the individual frames in the sync packet into one large block

        Currently just stacking all frames vertically, but this should be expanded on the 
        the future to allow wrapping to a more square shape
        """

        self.new_sync_packet_notice.get()
        self.current_sync_packet = self.synchronizer.current_sync_packet
        
        thumbnail_frames = {} 
        for port in self.ports:
            frame_packet = self.current_sync_packet.frame_packets[port]
            raw_frame = self.get_frame_or_blank(frame_packet)
            square_frame = self.resize_to_square(raw_frame)
            rotated_frame = self.apply_rotation(square_frame,port)
            
            # put the port number on the top of the frame
            text_frame = cv2.putText(rotated_frame,
                                str(port),
                                (int(rotated_frame.shape[1]/2), int(self.single_frame_height / 4)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0, 0, 255),
                                thickness=2,
                            )
            
            thumbnail_frames[port] = text_frame
                    
        frame_rows = [] 
        current_row = None
        current_row_length = 0
        frames_added = 0 
        frames_remaining = len(self.ports)
        for port,frame in thumbnail_frames.items():
            # for column in range(self.frame_columns):
            if current_row is None:
                current_row = frame
            else:
                current_row = np.hstack([current_row,frame])  
            current_row_length +=1
            frames_remaining -=1

            if frames_remaining ==0:
                # pad with blanks
                while current_row_length < self.frame_columns:
                    current_row = np.hstack([current_row,self.get_frame_or_blank(None)]) 
                    current_row_length += 1

            if current_row_length == self.frame_columns:
                frame_rows.append(current_row)            
                current_row = None
                current_row_length = 0
        
        mega_frame = None
        for row in frame_rows:
            if mega_frame is None:
                mega_frame = row
            else:
                mega_frame = np.vstack([mega_frame,row]) 
                         
         
        return mega_frame

class RecordingFrameEmitter(QThread):
    ImageBroadcast = pyqtSignal(QImage)
    dropped_fps = pyqtSignal(dict)
    
    def __init__(self, recording_frame_builder:RecordingFrameBuilder):
        
        super(RecordingFrameEmitter,self).__init__()
        self.recording_frame_builder = recording_frame_builder
        logger.info("Initiated recording frame emitter")        
        self.keep_collecting = Event() 
        
    def run(self):

        self.keep_collecting.set()
        
        while self.keep_collecting.is_set():
            # that that it is important to make sure that this signal is sent only once
            # to avoid multiple calibration attempts 
                      
            recording_frame = self.recording_frame_builder.get_recording_frame()

            if recording_frame is not None:
                image = cv2_to_qlabel(recording_frame)
                self.ImageBroadcast.emit(image)
                self.dropped_fps.emit(self.recording_frame_builder.synchronizer.dropped_fps)

        logger.info("Stereoframe emitter run thread ended...") 
            


def get_empty_pairs(board_counts, min_threshold):
    empty_pairs = [key for key, value in board_counts.items() if value < min_threshold]
    return empty_pairs

def resize(image, new_height):
    (current_height, current_width) = image.shape[:2]
    ratio = new_height / float(current_height)
    dim = (int(current_width * ratio), new_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized
        
        
        
def cv2_to_qlabel(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    qt_frame = QImage(
        image.data,
        image.shape[1],
        image.shape[0],
        QImage.Format.Format_RGB888,
    )
    return qt_frame

# Trying to get away from these F5 tests and move toward working scripts in /dev that can 
# more easily be reconfigured into /tests  
# if __name__ == "__main__":
        # App = QApplication(sys.argv)

        # session_path = Path(__root__, "dev", "sample_sessions", "post_optimization")

        # session = Session(session_path)
        # session.load_cameras()
        # session.load_streams()
        
        # toggle off tracking for max frame rate
        # for port, stream in session.streams.items():
        #     stream.track_points.clear()
            
        # session.adjust_resolutions()
        # syncr = Synchronizer(session.streams, fps_target=24)

        # frame_builder = RecordingFrameBuilder(syncr)
        
        # while True:
        #     recording_frame = frame_builder.get_recording_frame()
        #     cv2.imshow("Recording Frame", recording_frame)
            
        #     key = cv2.waitKey(1)

        #     if key == ord("q"):
        #         cv2.destroyAllWindows()
        #         break

        # sys.exit(App.exec())

        # App = QApplication(sys.argv)


        # session = Session(session_path)
        # session.load_cameras()
        # session.load_streams()
        # session.adjust_resolutions()


        # recording_dialog = RecordingWidget(session)
        # recording_dialog.show()

        # sys.exit(App.exec())