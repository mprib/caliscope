

import caliscope.logger
from time import sleep
import math
from threading import Event
import numpy as np

import cv2
from PySide6.QtCore import Signal,Slot, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QWidget,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from caliscope.cameras.synchronizer import Synchronizer
from caliscope.packets import FramePacket
from caliscope.controller import Controller

logger = caliscope.logger.get(__name__)
 
class ExtrinsicPlaybackWidget(QWidget):
     
    def __init__(self,controller:Controller):

        super(ExtrinsicPlaybackWidget, self).__init__()
        self.controller = controller
        self.synchronizer:Synchronizer = self.controller.synchronizer
        self.ports = self.synchronizer.ports

        # need to let synchronizer spin up before able to display frames
        while not hasattr(controller.synchronizer, "current_sync_packet"):
            sleep(.25)
        # create tools to build and emit the displayed frame
        # self.unpaired_frame_builder = FramePrepper(self.synchronizer)
        self.thumbnail_emitter = FrameDictionaryEmitter(self.synchronizer)
        self.thumbnail_emitter.start()

        # all video output routed to qlabels stored in a dictionariy 
        # make it as square as you can get it
        self.playback_displays = {str(port):QLabel() for port in self.ports}
        # self.recording_frame_display = QLabel()
        
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
        for port in sorted(self.ports):
            frame_grid.addWidget(self.playback_displays[str(port)], row,column)
            
            # update row and column for next iteration
            if column >= grid_columns-1:
                # start fresh on next row
                column = 0
                row += 1
            else:
                column += 1
            
        frame_display_layout = QHBoxLayout()
        frame_display_layout.addStretch(1)
        frame_display_layout.addLayout(frame_grid)
        frame_display_layout.addStretch(1)
        self.layout().addLayout(frame_display_layout)


    def connect_widgets(self):
    
        self.thumbnail_emitter.ThumbnailImagesBroadcast.connect(self.ImageUpdateSlot)
        self.frame_rate_spin.valueChanged.connect(self.controller.set_active_mode_fps)
        self.thumbnail_emitter.dropped_fps.connect(self.update_dropped_fps)
        self.start_stop.clicked.connect(self.toggle_start_stop)
        self.controller.qt_signaler.recording_complete_signal.connect(self.on_recording_complete)
        
         
    @Slot(dict) 
    def ImageUpdateSlot(self, q_image_dict:dict):
        logger.debug("About to get qpixmap from qimage")
        for port, thumbnail in q_image_dict.items():
            qpixmap = QPixmap.fromImage(thumbnail)
            logger.debug("About to set qpixmap to display")
            self.playback_displays[port].setPixmap(qpixmap)
            logger.debug("successfully set display")
        

class FrameDictionaryEmitter(QThread):
    ThumbnailImagesBroadcast = Signal(dict)
    dropped_fps = Signal(dict)
    
    def __init__(self, synchronizer: Synchronizer, single_frame_height=300):
        
        super(FrameDictionaryEmitter,self).__init__()
        self.single_frame_height = single_frame_height
        self.synchronizer = synchronizer
        logger.info("Initiated recording frame emitter")        
        self.keep_collecting = Event() 
       
    def run(self):

        self.keep_collecting.set()
        
        while self.keep_collecting.is_set():
            logger.debug("About to get next recording frame")
            # recording_frame = self.unpaired_frame_builder.get_recording_frame()
            
            logger.debug("Referencing current sync packet in synchronizer")
            self.current_sync_packet = self.synchronizer.current_sync_packet
        
            thumbnail_qimage = {} 
            for port in self.synchronizer.ports:
                frame_packet = self.current_sync_packet.frame_packets[port]
                rotation_count = self.synchronizer.streams[port].camera.rotation_count

                text_frame = frame_packet_2_thumbnail(frame_packet, rotation_count, self.single_frame_height, port) 
                q_image = cv2_to_qimage(text_frame)
                thumbnail_qimage[str(port)] = q_image
            
            self.ThumbnailImagesBroadcast.emit(thumbnail_qimage)
            
            dropped_fps_dict = {str(port):dropped for port, dropped in self.synchronizer.dropped_fps.items()}
            self.dropped_fps.emit(dropped_fps_dict)
        logger.info("Recording thumbnail emitter run thread ended...") 
    
def frame_packet_2_thumbnail(frame_packet:FramePacket, rotation_count:int, edge_length:int, port:int):
    raw_frame = get_frame_or_blank(frame_packet, edge_length)
    # port = frame_packet.port        

    # raw_frame = self.get_frame_or_blank(None)
    square_frame = resize_to_square(raw_frame, edge_length)
    rotated_frame = apply_rotation(square_frame, rotation_count)
    flipped_frame = cv2.flip(rotated_frame, 1)

    # put the port number on the top of the frame
    text_frame = cv2.putText(flipped_frame,
                        str(port),
                        (int(flipped_frame.shape[1]/2), int(edge_length / 4)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2,
                    )

    return text_frame        


def get_frame_or_blank(frame_packet: FramePacket, edge_length):
    """Synchronization issues can lead to some frames being None among
    the synched frames, so plug that with a blank frame"""

    if frame_packet is None:
        logger.debug("plugging blank frame data")
        frame = np.zeros((edge_length, edge_length, 3), dtype=np.uint8)
    else:
        frame = frame_packet.frame_with_points

    return frame

def resize_to_square(frame, edge_length):
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

    frame = resize(frame, new_height=edge_length)
    return frame

def resize(image, new_height):
    (current_height, current_width) = image.shape[:2]
    ratio = new_height / float(current_height)
    dim = (int(current_width * ratio), new_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def apply_rotation(frame, rotation_count):
    if rotation_count == 0:
        pass
    elif rotation_count in [1, -3]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_count in [2, -2]:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_count in [-1, 3]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame

     
def prep_img_for_qpixmap(image:np.ndarray):
    """
    qpixmap needs dimensions divisible by 4 and without that weird things happen.
    """
    if image.shape[1] % 4 != 0:  # If the width of the row isn't divisible by 4
        padding_width = 4 - (image.shape[1] % 4)  # Calculate how much padding is needed
        padding = np.zeros((image.shape[0], padding_width, image.shape[2]), dtype=image.dtype)  # Create a black image of the required size
        image = np.hstack([image, padding])  # Add the padding to the right of the image

    return image


def cv2_to_qimage(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    qt_frame = QImage(
        image.data,
        image.shape[1],
        image.shape[0],
        QImage.Format.Format_RGB888,
    )

    return qt_frame
