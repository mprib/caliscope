from pyxy3d.cameras.camera_array import CameraArray


class RealTimeTriangulator:
    
    def __init__(self,camera_array:CameraArray):
        self.camera_array = CameraArray