


import sys
from pathlib import Path
from threading import Thread

import cv2


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.cameras.camera import Camera


class CameraArray:

    def __init__(self, count=2):

        self.target_cam_count = count
        self.cameras = {}

        self.find_cameras()
    def try_port(self, port):

        try:
            self.cameras[port] = Camera(port)
            print(f"Camera found at port {port}")
            self.target_cam_count = self.target_cam_count-1
        except:
            print(f"No camera at port {port}")
            pass
            
    @property
    def cam_count(self):
        len(self.cameras)


    def find_cameras(self):
        try_port_threads = []
        for test in range(0,self.target_cam_count+5):
            try_port(test)

if __name__ == "__main__":
    test_array = CameraArray(2)
