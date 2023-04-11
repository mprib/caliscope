

import pyxy3d.logger
logger = pyxy3d.logger.get(__name__)

import sys
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

# Append main repo to top of path to allow import of backend
from pyxy3d.session import Session
from pyxy3d.gui.stereoframe.stereo_frame_builder import StereoFrameBuilder
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d import __root__
from pyxy3d.gui.qt_logger import QtLogger
from pyxy3d.gui.widgets import NavigationBarBackFinish

# the boards needed to before a pair could be used to bridge pairs without common corners
MIN_THRESHOLD_FOR_EARLY_CALIBRATE = 5


class RecordingWidget(QWidget):
     
    def __init__(self,session:Session):

        super(RecordingWidget, self).__init__()
        self.session = session
        self.synchronizer:Synchronizer = self.session.get_synchronizer()

        # create tools to build and emit the displayed frame
        self.frame_builder = StereoFrameBuilder(self.synchronizer)
        self.frame_emitter = RecordingFrameEmitter(self.frame_builder)
        self.frame_emitter.start()

        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setValue(self.synchronizer.get_fps_target())
        
        self.recording_frame_display = QLabel()
        
        self.place_widgets()
        self.connect_widgets()        


    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        
        pass

    def connect_widgets(self):
        pass    
    
        
    def ImageUpdateSlot(self, q_image):
        self.recording_frame_display.resize(self.recording_frame_display.sizeHint())

        qpixmap = QPixmap.fromImage(q_image)
        self.recording_frame_display.setPixmap(qpixmap)
        

class RecordingFrameEmitter(QThread):
    ImageBroadcast = pyqtSignal(QImage)
    
    def __init__(self, stereoframe_builder:StereoFrameBuilder):
        
        super(RecordingFrameEmitter,self).__init__()
        self.stereoframe_builder = stereoframe_builder
        logger.info("Initiated frame emitter")        
        self.keep_collecting = Event() 
        self.collection_complete = False
        
    def run(self):

        self.keep_collecting.set()
        self.collection_complete = False

        possible_to_initialize = False
        
        while self.keep_collecting.is_set():
            # that that it is important to make sure that this signal is sent only once
            # to avoid multiple calibration attempts 
            if len(self.stereoframe_builder.stereo_list) == 0 and not self.collection_complete:
                logger.info("Signalling that calibration data is fully collected.")
                self.collection_complete = True
                self.calibration_data_collected.emit()
        
                # break
            
            if not possible_to_initialize:
                # check to see if it is now
                if self.stereoframe_builder.possible_to_initialize_array(MIN_THRESHOLD_FOR_EARLY_CALIBRATE):
                    logger.info("Signaling that it is possible to initialize array based on collected data.")
                    possible_to_initialize = True
                    self.possible_to_initialize_array.emit()
                      
            stereo_frame = self.stereoframe_builder.get_stereo_frame()

            if stereo_frame is not None:
                image = cv2_to_qlabel(stereo_frame)
                self.ImageBroadcast.emit(image)

        logger.info("Stereoframe emitter run thread ended...") 
            
    # def stop(self):
        # self.keep_collecting.clear() 


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
            
        mega_frame = None
        for port,frame in thumbnail_frames.items():
            if mega_frame is None:
                mega_frame = frame
            else:
                mega_frame = np.vstack([mega_frame,frame])  
            
        return mega_frame



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

    
if __name__ == "__main__":
        App = QApplication(sys.argv)

        session_path = Path(__root__, "dev", "sample_sessions", "post_optimization")

        session = Session(session_path)
        session.load_cameras()
        session.load_streams()
        session.adjust_resolutions()
        syncr = Synchronizer(session.streams, fps_target=12)

        frame_builder = RecordingFrameBuilder(syncr)
        
        while True:
            recording_frame = frame_builder.get_recording_frame()
            cv2.imshow("Recording Frame", recording_frame)
            
            key = cv2.waitKey(1)

            if key == ord("q"):
                cv2.destroyAllWindows()
                break

        sys.exit(App.exec())
