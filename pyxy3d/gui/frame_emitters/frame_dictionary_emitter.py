import pyxy3d.logger
import numpy as np

from threading import Event
from queue import Queue

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import pyxy3d.calibration.draw_charuco as draw_charuco
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.interface import FramePacket, SyncPacket
from pyxy3d.cameras.camera_array import CameraData
from pyxy3d.gui.frame_emitters.tools import resize_to_square, apply_rotation, cv2_to_qlabel

logger = pyxy3d.logger.get(__name__)


class FrameDictionaryEmitter(QThread):
    # establish signals that will be displayed within the GUI
    FramesBroadcast = Signal(dict[QPixmap])
    # GridCountBroadcast = Signal(int)
    # FrameIndexBroadcast = Signal(int, int)

    def __init__(self, synchronizer: Synchronizer,all_camera_data:dict[CameraData], pixmap_edge_length=500):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(FrameDictionaryEmitter, self).__init__()

        self.synchronizer = synchronizer
        self.streams = self.synchronizer.streams
        self.all_camera_data = all_camera_data

        self.sync_packet_q = Queue()
        self.synchronizer.subscribe(self.sync_packet_q)
        self.pixmap_edge_length = pixmap_edge_length
        self.keep_collecting = Event()

    def run(self):
        self.keep_collecting.set()

        while self.keep_collecting.is_set():
            # Grab a frame from the queue and broadcast to displays
            # self.monocalibrator.grid_frame_ready_q.get()
            logger.info("Getting frame packet from queue")
            sync_packet = self.sync_packet_q.get()
            emitted_dict = {}
            for port, frame_packet in sync_packet.frame_packets.items():
                rotation_count = self.streams[port].rotation_count
                frame = frame_packet.frame_with_points
                frame = resize_to_square(frame)
                frame = apply_rotation(frame, rotation_count)
                image = cv2_to_qlabel(frame)
                pixmap = QPixmap.fromImage(image)

                if self.pixmap_edge_length:
                    pixmap = pixmap.scaled(
                        int(self.pixmap_edge_length),
                        int(self.pixmap_edge_length),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                emitted_dict[str(port)] = frame               

            self.FramesBroadcast.emit(emitted_dict) 

        logger.info(
            f"Thread loop within frame emitter at port {self.synchronizer.port} successfully ended"
        )

    def stop(self):
        self.keep_collecting = False
        self.quit()
