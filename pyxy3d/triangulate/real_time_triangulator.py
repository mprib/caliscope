from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.cameras.synchronizer import Synchronizer


class RealTimeTriangulator:
    
    def __init__(self,camera_array:CameraArray, synchronizer:Synchronizer):
        self.camera_array = camera_array
        self.synchronizer = synchronizer