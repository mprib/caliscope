# In calibration/protocol.py
from typing import Protocol
from caliscope.post_processing.point_data import ImagePoints
from caliscope.cameras.camera_array import CameraArray
from caliscope.calibration.stereopairs import StereoPairs


class StereoCalibratorProtocol(Protocol):
    def calibrate(self, image_points: ImagePoints, camera_array: CameraArray) -> StereoPairs: ...
