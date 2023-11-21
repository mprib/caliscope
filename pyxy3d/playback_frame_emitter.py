import pyxy3d.logger
import numpy as np

from datetime import datetime
from pathlib import Path
from time import sleep
from threading import Event
from queue import Queue

import cv2
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QImage, QPixmap
import pyxy3d.calibration.draw_charuco as draw_charuco
from pyxy3d.calibration.monocalibrator import MonoCalibrator
from pyxy3d.calibration.intrinsic_calibrator import IntrinsicCalibrator
from pyxy3d.recording.recorded_stream import RecordedStream

logger = pyxy3d.logger.get(__name__)


class PlaybackFrameEmitter(QThread):
    # establish signals that will be displayed within the GUI
    ImageBroadcast = Signal(int, QPixmap)
    GridCountBroadcast = Signal(int)
    FrameIndexBroadcast = Signal(int, int)

    def __init__(self, recorded_stream: RecordedStream, pixmap_edge_length=500):
        # pixmap_edge length is from the display window. Keep the display area
        # square to keep life simple.
        super(PlaybackFrameEmitter, self).__init__()
        self.stream = recorded_stream
        self.port = self.stream.port

        self.frame_packet_q = Queue()
        self.stream.subscribe(self.frame_packet_q)
        self.pixmap_edge_length = pixmap_edge_length
        self.undistort = False
        self.keep_collecting = Event()
        self.initialize_grid_capture_history()

    def initialize_grid_capture_history(self):
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
            logger.info("Getting frame packet from queue")
            frame_packet = self.frame_packet_q.get()
            self.frame = frame_packet.frame_with_points

            logger.info(f"Frame size is {self.frame.shape}")
            logger.info(
                f"Grid Capture History size is {self.grid_capture_history.shape}"
            )
            self.frame = cv2.addWeighted(self.frame, 1, self.grid_capture_history, 1, 0)

            self._apply_undistortion()
            
            # cv2.imshow("emitted frame", self.frame)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     break
            

            logger.info(f"Frame size is {self.frame.shape} following undistortion")
            self.frame = resize_to_square(self.frame)

            self.apply_rotation()

            image = self.cv2_to_qlabel(self.frame)
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
        self.keep_collecting = False
        self.quit()

    def cv2_to_qlabel(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # FlippedImage = cv2.flip(Image, 1)

        qt_frame = QImage(
            Image.data,
            Image.shape[1],
            Image.shape[0],
            QImage.Format.Format_RGB888,
        )
        return qt_frame

    def apply_rotation(self):
        logger.info(f"Current rotation count is {self.stream.rotation_count}")
        if self.stream.rotation_count == 0:
            pass
        elif self.stream.rotation_count in [1, -3]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.stream.rotation_count in [2, -2]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        elif self.stream.rotation_count in [-1, 3]:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def update_distortion_params(self, undistort, matrix, distortions):
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

            # Apply a safety margin to the scaling factors (e.g., 5%)
            self.scaling_factor = 0.95
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
        logger.info("Attempting to add to grid history")
        if len(ids) > 3:
            logger.info("enough points to add")
            self.grid_capture_history = draw_charuco.grid_history(
                self.grid_capture_history,
                ids,
                img_loc,
                self.connected_points,
            )
        else:
            logger.info("Not enough points....grid not added...")


def resize_to_square(frame):
    height = frame.shape[0]
    width = frame.shape[1]

    padded_size = max(height, width)

    height_pad = int((padded_size - height) / 2)
    width_pad = int((padded_size - width) / 2)
    pad_color = [0, 0, 0]
    pad_color = [100, 100, 100]

    frame = cv2.copyMakeBorder(
        frame,
        height_pad,
        height_pad,
        width_pad,
        width_pad,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    return frame
