import pyxy3d.logger
from time import sleep
import math

from PySide6.QtCore import Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QMainWindow,
    QWidget,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.gui.frame_emitters.frame_dictionary_emitter import FrameDictionaryEmitter
from pyxy3d.synchronized_stream_manager import SynchronizedStreamManager
from pyxy3d.controller import Controller

logger = pyxy3d.logger.get(__name__)


class SynchedFramesDisplay(QWidget):
    """
    This widget is not intended to have any interactive functionality at all and to only 
    provide a window to the user of the current landmark tracking
    
    This is why the primary input is the sync stream manager directly and not the controller 
    Apologies to Future Mac who is reading this and regretting my decisions.
    """

    def __init__(self, sync_stream_manager: SynchronizedStreamManager):
        super(SynchedFramesDisplay, self).__init__()
        self.sync_stream_manager = sync_stream_manager 
        self.synchronizer = self.sync_stream_manager.synchronizer
        self.ports = self.synchronizer.ports

        # need to let synchronizer spin up before able to display frames
        while not hasattr(self.synchronizer, "current_sync_packet"):
            sleep(0.25)

        # create tools to build and emit the displayed frame
        self.frame_dictionary_emitter = FrameDictionaryEmitter(
            self.synchronizer, self.sync_stream_manager.all_camera_data
        )
        self.frame_dictionary_emitter.start()

        self.dropped_fps_label = QLabel()

        # all video output routed to qlabels stored in a dictionariy
        # make it as square as you can get it
        self.recording_displays = {str(port): QLabel() for port in self.ports}

        self.place_widgets()
        self.connect_widgets()

        logger.info("Recording widget init complete")

    def place_widgets(self):
        self.setLayout(QVBoxLayout())

        dropped_fps_layout = QHBoxLayout()
        dropped_fps_layout.addStretch(1)
        dropped_fps_layout.addWidget(self.dropped_fps_label)
        dropped_fps_layout.addStretch(1)

        # self.layout().addLayout(dropped_fps_layout)

        # set teh layout for the frames to be mostly square
        frame_grid = QGridLayout()
        camera_count = len(self.ports)
        grid_columns = int(math.ceil(camera_count**0.5))
        row = 0
        column = 0
        for port in sorted(self.ports):
            frame_grid.addWidget(self.recording_displays[str(port)], row, column)

            # update row and column for next iteration
            if column >= grid_columns - 1:
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
        self.frame_dictionary_emitter.FramesBroadcast.connect(self.ImageUpdateSlot)
        self.frame_dictionary_emitter.dropped_fps.connect(self.update_dropped_fps)

    @Slot(dict)
    def update_dropped_fps(self, dropped_fps: dict):
        "Unravel dropped fps dictionary to a more readable string"
        text = "Rate of Frame Dropping by Port:    "
        for port, drop_rate in dropped_fps.items():
            text += f"{port}: {drop_rate:.0%}        "
        self.dropped_fps_label.setText(text)

    @Slot(dict)
    def ImageUpdateSlot(self, qpixmaps: dict):
        # logger.info(f"About to process q_image_dict: {q_image_dict}")
        for port, qpixmap in qpixmaps.items():
            # qpixmap = QPixmap.fromImage(qpixmap)
            logger.debug("About to set qpixmap to display")
            self.recording_displays[str(port)].setPixmap(qpixmap)
            logger.debug("successfully set display")
