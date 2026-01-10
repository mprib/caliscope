import logging
from queue import Queue
from threading import Event

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap

import caliscope.core.draw_charuco as draw_charuco
from caliscope.gui.camera_undistort_view import CameraUndistortView
from caliscope.gui.frame_emitters.tools import apply_rotation, cv2_to_qlabel, resize_to_square
from caliscope.recording.recorded_stream import RecordedStream

logger = logging.getLogger(__name__)


class PlaybackFrameEmitter(QThread):
    # establish signals that will be displayed within the GUI
    ImageBroadcast = Signal(int, QPixmap)
    GridCountBroadcast = Signal(int)
    FrameIndexBroadcast = Signal(int, int)

    def __init__(self, recorded_stream: RecordedStream, grid_history_q: Queue, pixmap_edge_length=500):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(PlaybackFrameEmitter, self).__init__()
        self.stream = recorded_stream
        self.port = self.stream.port

        self.frame_packet_q = Queue()
        self.grid_history_q = grid_history_q  # received a tuple of ids, img_loc

        self.stream.subscribe(self.frame_packet_q)
        self.pixmap_edge_length = pixmap_edge_length
        self._undistort_enabled = False
        self._undistort_view: CameraUndistortView | None = None
        self.keep_collecting = Event()
        self.initialize_grid_capture_history()

    def initialize_grid_capture_history(self):
        """
        The grid capture history is only used as a GUI element to display to the user
        the corners that have been collected and are being used in the calibration.

        It is not otherwise used in any calculations and needs to be re-initialized by
        the intrinsic stream manager whenever the intrinsic calibrators data is also
        re-initialized.
        """
        self.connected_points = self.stream.tracker.get_connected_points() if self.stream.tracker else set()
        width = self.stream.size[0]
        height = self.stream.size[1]
        channels = 3
        self.grid_capture_history = np.zeros((height, width, channels), dtype="uint8")

    def run(self):
        self.keep_collecting.set()

        while self.keep_collecting.is_set():
            # Grab a frame from the queue and broadcast to displays
            # self.monocalibrator.grid_frame_ready_q.get()
            logger.debug("Getting frame packet from queue")
            frame_packet = self.frame_packet_q.get()

            while self.grid_history_q.qsize() > 0:
                ids, img_loc = self.grid_history_q.get()
                self.add_to_grid_history(ids, img_loc)

            if not self.keep_collecting.is_set():
                break

            if frame_packet.frame is not None:  # stream end signal when None frame placed on out queue
                self.frame = frame_packet.frame_with_points

                logger.debug(f"Frame size is {self.frame.shape}")
                logger.debug(f"Grid Capture History size is {self.grid_capture_history.shape}")
                self.frame = cv2.addWeighted(self.frame, 1, self.grid_capture_history, 1, 0)

                self._apply_undistortion()

                logger.debug(f"Frame size is {self.frame.shape} following undistortion")
                self.frame = resize_to_square(self.frame)
                self.frame = apply_rotation(self.frame, self.stream.rotation_count)
                image = cv2_to_qlabel(self.frame)
                pixmap = QPixmap.fromImage(image)

                if self.pixmap_edge_length:
                    pixmap = pixmap.scaled(
                        int(self.pixmap_edge_length),
                        int(self.pixmap_edge_length),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                self.ImageBroadcast.emit(self.port, pixmap)
                self.FrameIndexBroadcast.emit(self.port, frame_packet.frame_index)

        logger.info(f"Thread loop within frame emitter at port {self.stream.port} successfully ended")

    def stop(self):
        logger.info(f"Beginning to shut down frame emitter at port {self.port}")
        self.stream.unsubscribe(self.frame_packet_q)
        self.keep_collecting.clear()
        self.frame_packet_q.put(-1)
        self.quit()

    def set_scale_factor(self, scaling_factor: float) -> None:
        """Update the display scale factor for undistortion."""
        if self._undistort_view is not None:
            self._undistort_view.set_scale_factor(scaling_factor)

    def set_undistort(self, undistort: bool) -> None:
        """Enable or disable undistortion for display."""
        camera = self.stream.camera
        if camera.matrix is None:
            logger.info(f"No camera matrix calculated yet at port {self.port}")
            return

        logger.info(f"Setting undistort={undistort} for frame emitter at port {self.port}")
        self._undistort_enabled = undistort

        # Create view on first enable (lazy initialization)
        if undistort and self._undistort_view is None:
            # stream.size is (width, height), but CameraUndistortView expects (height, width)
            h, w = self.stream.size[1], self.stream.size[0]
            self._undistort_view = CameraUndistortView(camera, (h, w))

    def _apply_undistortion(self) -> None:
        """Apply undistortion to current frame if enabled."""
        if self._undistort_enabled and self._undistort_view is not None:
            self.frame = self._undistort_view.undistort_frame(self.frame)

    def add_to_grid_history(self, ids, img_loc):
        """
        Note that the connected points here comes from the charuco tracker.
        This grid history is likely best tracked by the controller and
        a reference should be past to the frame emitter
        """
        # logger.info("Attempting to add to grid history")
        if len(ids) > 3:
            # logger.info("enough points to add")
            self.grid_capture_history = draw_charuco.grid_history(
                self.grid_capture_history,
                ids,
                img_loc,
                self.connected_points,
            )
        else:
            logger.info("Not enough points....grid not added...")
