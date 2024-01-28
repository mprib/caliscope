import caliscope.logger
from time import sleep
import math

from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QMainWindow,
    QWidget,
    QScrollArea,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from caliscope.cameras.synchronizer import Synchronizer
from caliscope.gui.frame_emitters.frame_dictionary_emitter import FrameDictionaryEmitter
from caliscope.synchronized_stream_manager import SynchronizedStreamManager

logger = caliscope.logger.get(__name__)


class SynchedFramesDisplay(QWidget):
    """
    This widget is not intended to have any interactive functionality at all and to only 
    provide a window to the user of the current landmark tracking
    
    This is why the primary input is the sync stream manager directly and not the controller 
    Apologies to Future Mac who is reading this and regretting my decisions.
    """

    def __init__(self, sync_stream_manager: SynchronizedStreamManager):
        super(SynchedFramesDisplay, self).__init__()

        self.setWindowTitle("Tracking Landmarks....")
        self.sync_stream_manager = sync_stream_manager 
        self.synchronizer = self.sync_stream_manager.synchronizer
        self.ports = self.synchronizer.ports


        while not hasattr(sync_stream_manager.synchronizer, "current_sync_packet"):
            logger.info("Waiting for synchronizer to have sync packet")
            sleep(0.25)

        # create tools to build and emit the displayed frame
        self.frame_dictionary_emitter = FrameDictionaryEmitter(
            self.synchronizer, self.sync_stream_manager.all_camera_data
        )
        self.frame_dictionary_emitter.start()

        # all video output routed to qlabels stored in a dictionariy
        # make it as square as you can get it
        self.recording_displays = {str(port): QLabel() for port in self.ports}

        self.place_widgets()
        self.connect_widgets()

        logger.info("Synched Frames Display init complete")

    def place_widgets(self):
        self.scroll_area = QScrollArea()
        grid_widget = QWidget()
        grid_widget.setLayout(QGridLayout())

        camera_count = len(self.ports)
        grid_columns = int(math.ceil(camera_count**0.5))
        row = 0
        column = 0
        for port in sorted(self.ports):
            grid_widget.layout().addWidget(self.recording_displays[str(port)], row, column)

            # update row and column for next iteration
            if column >= grid_columns - 1:
                # start fresh on next row
                column = 0
                row += 1
            else:
                column += 1

        grid_widget.setLayout(grid_widget.layout())
        padded_grid_layout = QHBoxLayout()
        padded_grid_layout.addStretch(1)
        padded_grid_layout.addWidget(grid_widget)
        padded_grid_layout.addStretch(1)

        scroll_viewport = QWidget()
        scroll_viewport.setLayout(padded_grid_layout)

        #Scroll Area Properties
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(scroll_viewport)
        # self.setCentralWidget(self.scroll_area)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll_area)
    
    def connect_widgets(self):
        self.frame_dictionary_emitter.FramesBroadcast.connect(self.ImageUpdateSlot)
        # self.frame_dictionary_emitter.dropped_fps.connect(self.update_dropped_fps)
        self.frame_dictionary_emitter.close_window.connect(self.close)

    # @Slot(dict)
    # def update_dropped_fps(self, dropped_fps: dict):
    #     "Unravel dropped fps dictionary to a more readable string"
    #     text = "Rate of Frame Dropping by Port:    "
    #     for port, drop_rate in dropped_fps.items():
    #         text += f"{port}: {drop_rate:.0%}        "
    #     self.dropped_fps_label.setText(text)

    @Slot(dict)
    def ImageUpdateSlot(self, qpixmaps: dict):
        # logger.info(f"About to process q_image_dict: {q_image_dict}")
        for port, qpixmap in qpixmaps.items():
            # qpixmap = QPixmap.fromImage(qpixmap)
            logger.debug("About to set qpixmap to display")
            self.recording_displays[str(port)].setPixmap(qpixmap)
            logger.debug("successfully set display")

    