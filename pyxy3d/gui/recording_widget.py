

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import copy
import sys
from time import perf_counter, sleep
import math
from pathlib import Path
from threading import Thread, Event
import numpy as np
from queue import Queue
from enum import Enum

import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QGridLayout,
    QApplication,
    QSizePolicy,
    QWidget,
    QSpinBox,
    QScrollArea,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QLineEdit,
    QDialog,
    QGroupBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from pyxy3d.session.session import Session, SessionMode
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.recording.video_recorder import VideoRecorder
from pyxy3d.configurator import Configurator


# Whatever the target frame rate, the GUI will only display a portion of the actual frames
# this is done to cut down on computational overhead. 
RENDERED_FPS = 6


class NextRecordingActions(Enum):
    StartRecording = "Start Recording"
    StopRecording = "Stop Recording"
    AwaitSave = "--Saving Frames--"
    
 
class RecordingWidget(QWidget):
     
    def __init__(self,session:Session):

        super(RecordingWidget, self).__init__()
        self.session = session
        self.synchronizer:Synchronizer = self.session.synchronizer
        self.ports = self.synchronizer.ports

        # need to let synchronizer spin up before able to display frames
        # while not hasattr(session.synchronizer, "current_sync_packet"):
            # sleep(.5)
        # create tools to build and emit the displayed frame
        self.unpaired_frame_builder = UnpairedFrameBuilder(self.synchronizer)
        self.unpaired_frame_emitter = UnpairedFrameEmitter(self.unpaired_frame_builder)
        self.unpaired_frame_emitter.start()

        self.video_recorder = VideoRecorder(self.synchronizer)
        
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(self.session.fps_recording)

        self.next_action = NextRecordingActions.StartRecording
        self.start_stop = QPushButton(self.next_action.value)
        self.destination_label = QLabel("Recording Destination:")
        self.recording_directory = QLineEdit(self.get_next_recording_directory())
        
        self.dropped_fps_label = QLabel()
        
        # all video output routed to qlabels stored in a dictionariy 
        # make it as square as you can get it
        self.recording_displays = {port:QLabel() for port in self.ports}
        # self.recording_frame_display = QLabel()
        
        self.place_widgets()
        self.connect_widgets()        
        self.update_btn_eligibility()
        logger.info("Recording widget init complete")
    
    def update_btn_eligibility(self):
        if self.session.is_recording_eligible():
            self.start_stop.setEnabled(True)
            logger.info("Record button eligibility updated: Eligible")
        else:
            self.start_stop.setEnabled(False)
            logger.info("Record button eligibility updated: Not Eligible")

            

    def get_next_recording_directory(self):

        folders = [item.name for item in self.session.path.iterdir() if item.is_dir()]
        recording_folders = [folder for folder in folders if folder.startswith("recording_")]
        recording_counts = [folder.split("_")[1] for folder in recording_folders]
        recording_counts = [int(rec_count) for rec_count in recording_counts if rec_count.isnumeric()]

        if len(recording_counts) == 0:
            next_directory = "recording_1"
        
        else:
            next_directory = "recording_" + str(max(recording_counts)+1)
       
        return next_directory 
        
        

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
        self.record_controls.layout().addWidget(self.destination_label)
        self.record_controls.layout().addWidget(self.recording_directory)

        self.layout().addWidget(self.record_controls)

        dropped_fps_layout = QHBoxLayout()
        dropped_fps_layout.addStretch(1)
        dropped_fps_layout.addWidget(self.dropped_fps_label)
        dropped_fps_layout.addStretch(1)

        self.layout().addLayout(dropped_fps_layout)

        camera_count = len(self.ports)
        grid_columns = int(math.ceil(camera_count**.5))
        grid_rows = int(math.ceil(camera_count/grid_columns))

        frame_grid = QGridLayout()
        row = 0
        column = 0        
        for port, qlabel in self.recording_displays.items():
            if column > grid_columns:
                # start fresh on next row
                column = 0
                row += 1
            else:
                frame_grid.addWidget(qlabel, row,column)
                column += 1
            
        frame_display_layout = QHBoxLayout()
        frame_display_layout.addStretch(1)
        frame_display_layout.addLayout(frame_grid)
        frame_display_layout.addStretch(1)
        self.layout().addLayout(frame_display_layout)


    def connect_widgets(self):
    
        self.unpaired_frame_emitter.ImageBroadcast.connect(self.ImageUpdateSlot)
        self.frame_rate_spin.valueChanged.connect(self.session.set_active_mode_fps)
        self.unpaired_frame_emitter.dropped_fps.connect(self.update_dropped_fps)
        self.start_stop.clicked.connect(self.toggle_start_stop)
        self.session.qt_signaler.recording_complete_signal.connect(self.on_recording_complete)

    def toggle_start_stop(self):
        logger.info("Start/Stop Recording Toggled...")
        if self.next_action == NextRecordingActions.StartRecording:
            self.next_action = NextRecordingActions.StopRecording
            self.start_stop.setText(self.next_action.value)
            self.recording_directory.setEnabled(False)

            logger.info("Initiate recording")
            recording_path:Path = Path(self.session.path, self.recording_directory.text()) 
            self.session.start_recording(recording_path)

        elif self.next_action == NextRecordingActions.StopRecording:
            self.start_stop.setEnabled(False)
            # need to wait for session to signal that recording is complete
            self.next_action = NextRecordingActions.AwaitSave
            # self.start_stop.setText("HELLO")
            self.start_stop.setText(self.next_action.value)
            logger.info("Stop recording and initiate final save of file") 
            self.session.stop_recording()

            next_recording = self.get_next_recording_directory()
            self.recording_directory.setEnabled(True)
            self.recording_directory.setText(next_recording)
            logger.info(f"successfully reset text and renamed recording directory to {next_recording}")

        elif self.next_action == NextRecordingActions.AwaitSave:
            logger.info("recording button toggled while awaiting save")
            
    def on_recording_complete(self):
        logger.info("Recording complete signal received...updating next action and button")
        self.next_action = NextRecordingActions.StartRecording
        self.start_stop.setText(self.next_action.value)
        logger.info("Enabling start/stop recording button")
        self.start_stop.setEnabled(True)
        logger.info("Successfully enabled start/stop recording button")
        # pass
        
                    
    def update_dropped_fps(self, dropped_fps:dict):
        "Unravel dropped fps dictionary to a more readable string"
        text = "Rate of Frame Dropping by Port:    "
        for port, drop_rate in dropped_fps.items():
            text += f"{port}: {drop_rate:.0%}        "

        self.dropped_fps_label.setText(text)
         
    @pyqtSlot(QImage) 
    def ImageUpdateSlot(self, q_image:QImage):
        logger.info("About to get qpixmap from qimage")
        qpixmap = QPixmap.fromImage(q_image)
        logger.info("About to set qpixmap to display")
        self.recording_frame_display.setPixmap(qpixmap)
        logger.info("successfully set display")
        
class UnpairedFrameBuilder(QObject):
    def __init__(self, synchronizer: Synchronizer, single_frame_height=100):
        self.synchronizer = synchronizer 
        self.single_frame_height = single_frame_height

        self.ports = []        
        for port, stream in self.synchronizer.streams.items():
            self.ports.append(port)
        self.ports.sort()
        
        # make it as square as you can get it
        camera_count = len(self.ports)
        self.frame_columns = int(math.ceil(camera_count**.5))

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
        # rotation_count = self.rotation_counts[port]
        rotation_count = self.synchronizer.streams[port].camera.rotation_count
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
        It takes the sync packet that is currently sitting in the synchronizer as the input
        """
        # update the wait milestones used by the frame emitter in the event that 
        # the target fps of the synchronizer has changed.
        # logger.info(f"Begin process of getting recording frame and rendering at {self.rendered_fps}")

        logger.info("Referencing current sync packet in synchronizer")
        self.current_sync_packet = self.synchronizer.current_sync_packet
        
        thumbnail_frames = {} 
        for port in self.ports:
            frame_packet = self.current_sync_packet.frame_packets[port]
            raw_frame = self.get_frame_or_blank(frame_packet)
            # raw_frame = self.get_frame_or_blank(None)
            square_frame = self.resize_to_square(raw_frame)
            rotated_frame = self.apply_rotation(square_frame,port)
            flipped_frame = cv2.flip(rotated_frame, 1)
            
            # put the port number on the top of the frame
            text_frame = cv2.putText(flipped_frame,
                                str(port),
                                (int(flipped_frame.shape[1]/2), int(self.single_frame_height / 4)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0, 0, 255),
                                thickness=2,
                            )
            
            thumbnail_frames[port] = text_frame
                    
        # Begin to assemble the individual frames into a grid
        frame_rows = [] 
        current_row = None # no frames laid down yet
        current_row_length = 0

        for port,frame in thumbnail_frames.items():
            if current_row is None:
                current_row = frame
                current_row_length += 1
            else:
                current_row = np.hstack([current_row, frame])  
                current_row_length += 1
                if current_row_length == self.frame_columns:

                    # current_row = prep_img_for_qpixmap(current_row)
                    frame_rows.append(current_row)            
                    current_row = None
                    current_row_length = 0
            
        # After the loop
        if current_row is not None:
            while current_row_length < self.frame_columns:
                current_row = np.hstack([current_row, self.get_frame_or_blank(None)]) 

                current_row_length += 1

            # current_row = prep_img_for_qpixmap(current_row)
            frame_rows.append(current_row)


        mega_frame = None
        for row in frame_rows:
            if mega_frame is None:
                mega_frame = row
            else:
                mega_frame = np.vstack([mega_frame,row]) 
        logger.info("Returning megaframe")
        return mega_frame

class UnpairedFrameEmitter(QThread):
    ImageBroadcast = pyqtSignal(QImage)
    dropped_fps = pyqtSignal(dict)
    
    def __init__(self, unpaired_frame_builder:UnpairedFrameBuilder):
        
        super(UnpairedFrameEmitter,self).__init__()
        self.unpaired_frame_builder = unpaired_frame_builder
        logger.info("Initiated recording frame emitter")        
        self.keep_collecting = Event() 
       
      
    # def wait_to_next_frame(self):
    #     """
    #     based on the next milestone time, return the time needed to sleep so that
    #     a frame read immediately after would occur when needed
    #     """
    #     logger.info("Begin wait to next frame")
    #     time = perf_counter()
    #     fractional_time = time % 1
    #     all_wait_times = self.unpaired_frame_builder.milestones - fractional_time
    #     future_wait_times = all_wait_times[all_wait_times > 0]

    #     if len(future_wait_times) == 0:
    #         wait =  1 - fractional_time
    #     else:
    #         wait =  future_wait_times[0]
        
    #     logger.info(f"Begin waiting for {wait}")
    #     sleep(wait)
        
        
    def run(self):

        self.keep_collecting.set()
        
        while self.keep_collecting.is_set():
            logger.info("About to get next recording frame")
            # self.wait_to_next_frame()
            sleep(1/RENDERED_FPS)
            recording_frame = self.unpaired_frame_builder.get_recording_frame()
            if recording_frame is not None:
                logger.info(f"Emitting frame of size {recording_frame.shape}")
                image = cv2_to_qlabel(recording_frame)
                # cv2.imshow("Recording Frame", recording_frame)
                # cv2.waitKey(0)
                self.ImageBroadcast.emit(image)
                self.dropped_fps.emit(self.unpaired_frame_builder.synchronizer.dropped_fps)
            else:
                logger.info("Recording frame is none...")
        logger.info("Stereoframe emitter run thread ended...") 
            
def prep_img_for_qpixmap(image:np.ndarray):
    """
    qpixmap needs dimensions divisible by 4 and without that weird things happen.
    """
    if image.shape[1] % 4 != 0:  # If the width of the row isn't divisible by 4
        padding_width = 4 - (image.shape[1] % 4)  # Calculate how much padding is needed
        padding = np.zeros((image.shape[0], padding_width, image.shape[2]), dtype=image.dtype)  # Create a black image of the required size
        image = np.hstack([image, padding])  # Add the padding to the right of the image

    return image

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


def launch_recording_widget(session_path):
            config = Configurator(session_path)
            session = Session(config)
            # session.load_stream_tools()
            # session._adjust_resolutions()
            session.set_mode(SessionMode.Recording)
            App = QApplication(sys.argv)
            recording_dialog = RecordingWidget(session)
            recording_dialog.show()

            sys.exit(App.exec())