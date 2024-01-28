import cv2
import numpy as np
from PySide6.QtGui import QImage
import caliscope.logger

logger = caliscope.logger.get(__name__)

def resize_to_square(frame):
    height = frame.shape[0]
    width = frame.shape[1]

    padded_size = max(height, width)

    height_pad = int((padded_size - height) / 2)
    width_pad = int((padded_size - width) / 2)
    pad_color = [0, 0, 0]
    # pad_color = [100, 100, 100]

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

def apply_rotation(frame, rotation_count:int):
    if rotation_count == 0:
        pass
    elif rotation_count in [1, -3]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_count in [2, -2]:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_count in [-1, 3]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame


def cv2_to_qlabel(frame):
    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # FlippedImage = cv2.flip(Image, 1)

    qt_frame = QImage(
        Image.data,
        Image.shape[1],
        Image.shape[0],
        QImage.Format.Format_RGB888,
    )
    return qt_frame

