import caliscope.logger
import numpy as np

from threading import Event
from queue import Queue

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import caliscope.calibration.draw_charuco as draw_charuco
from caliscope.recording.recorded_stream import RecordedStream
from caliscope.gui.frame_emitters.tools import resize_to_square, apply_rotation, cv2_to_qlabel

logger = caliscope.logger.get(__name__)


class PlaybackFrameEmitter(QThread):
    # establish signals that will be displayed within the GUI
    ImageBroadcast = Signal(int, QPixmap)
    GridCountBroadcast = Signal(int)
    FrameIndexBroadcast = Signal(int, int)

    def __init__(self, recorded_stream: RecordedStream,grid_history_q:Queue,  pixmap_edge_length=500): 
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(PlaybackFrameEmitter, self).__init__()
        self.stream = recorded_stream
        self.port = self.stream.port
        
        # Apply a safety margin to the scaling factors (e.g., 5%)
        # used only when applying the undistortion
        self.scaling_factor = 1

        self.frame_packet_q = Queue()
        self.grid_history_q = grid_history_q  # received a tuple of ids, img_loc

        self.stream.subscribe(self.frame_packet_q)
        self.pixmap_edge_length = pixmap_edge_length
        self.undistort = False
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
        self.connected_points = self.stream.tracker.get_connected_points()
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
                ids,img_loc = self.grid_history_q.get()
                self.add_to_grid_history(ids, img_loc)

            if not self.keep_collecting.is_set():
                break

            if frame_packet.frame is not None:  # stream end signal when None frame placed on out queue
                self.frame = frame_packet.frame_with_points

                logger.debug(f"Frame size is {self.frame.shape}")
                logger.debug(
                    f"Grid Capture History size is {self.grid_capture_history.shape}"
                )
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
            


        logger.info(
            f"Thread loop within frame emitter at port {self.stream.port} successfully ended"
        )

    def stop(self):
        logger.info(f"Beginning to shut down frame emitter at port {self.port}")
        self.stream.unsubscribe(self.frame_packet_q)
        self.keep_collecting.clear()
        self.frame_packet_q.put(-1)
        self.quit()
        
    def set_scale_factor(self, scaling_factor):
        self.scaling_factor = scaling_factor

        if hasattr(self, "matrix") and hasattr(self, "distortions"):
            self.update_distortion_params(self.undistort,self.matrix, self.distortions)

    def update_distortion_params(self, undistort=None, matrix=None, distortions=None):
        if matrix is None:
            logger.info(f"No camera matrix calculated yet at port {self.port}")
        else:
            logger.info(
                f"Updating camera matrix and distortion parameters for frame emitter at port {self.port}"
            )
            
            self.undistort = undistort
            self.matrix = matrix
            self.distortions = distortions

            h, w = self.stream.size
            # h, w = original_image_size
            initial_new_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
                self.matrix, self.distortions, (w, h), 1
            )

            logger.info(f"Valid ROI is {valid_roi}")

            # Find extreme points and midpoints in the original image
            corners = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)
            midpoints = np.array([[w/2, 0], [w/2, h], [0, h/2], [w, h/2]], dtype=np.float32)
            extreme_points = np.vstack((corners, midpoints))

            # Undistort these points
            undistorted_points = cv2.undistortPoints(
                np.expand_dims(extreme_points, axis=1),
                self.matrix,
                self.distortions,
                P=initial_new_matrix,
            )

            # Find min/max x and y in undistorted points
            min_x = min(undistorted_points[:, 0, 0])
            max_x = max(undistorted_points[:, 0, 0])
            min_y = min(undistorted_points[:, 0, 1])
            max_y = max(undistorted_points[:, 0, 1])

            # # Calculate new image width and height
            # new_width = int(np.ceil(max_x - min_x))
            # new_height = int(np.ceil(max_y - min_y))

            # Calculate scaling factors
            scale_x = (max_x - min_x) / w
            scale_y = (max_y - min_y) / h

            scale_x *= self.scaling_factor
            scale_y *= self.scaling_factor

            # Adjust new image width and height
            adjusted_width = int(w * scale_x)
            adjusted_height = int(h * scale_y)

            logger.info(f"New image size for undistorted frame: {(adjusted_width,adjusted_height)}")
            # Now use new_width and new_height as your NewImageSize for undistortion
            # newImageSize = (new_width, new_height)
            self.new_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
                self.matrix, self.distortions, (w, h), 1, (adjusted_width, adjusted_height)
            )

    def _apply_undistortion(self):
        if self.undistort and self.matrix is not None:
            # Compute the optimal new camera matrix
            # Undistort the image
            self.frame = cv2.undistort(
                self.frame, self.matrix, self.distortions, None, self.new_matrix
            )

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


