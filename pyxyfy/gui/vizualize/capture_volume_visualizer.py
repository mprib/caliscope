import pyxyfy.logger

logger = pyxyfy.logger.get(__name__)

import math
import sys
import time
from pathlib import Path
from threading import Thread
from queue import Queue
import pandas as pd
import numpy as np
import pyqtgraph.opengl as gl

from random import random

from pyxyfy.cameras.camera_array import CameraData
from pyxyfy.gui.vizualize.camera_mesh import CameraMesh
from pyxyfy.cameras.camera_array_builder import CameraArray, CameraArrayBuilder
from pyxyfy.calibration.capture_volume.capture_volume import CaptureVolume

class CaptureVolumeVisualizer:
    def __init__(self, capture_volume:CaptureVolume = None, camera_array:CameraArray = None):
        if camera_array is not None and capture_volume is None:
            self.camera_array = camera_array
            self.point_estimates = None
        else:
            self.capture_volume = capture_volume
            self.camera_array = capture_volume.camera_array
            self.point_estimates = self.capture_volume.point_estimates

        self.current_frame = 0

        # constuct a scene
        self.scene = gl.GLViewWidget()
        self.scene.setCameraPosition(distance=4) # the scene camera, not a real Camera

        axis = gl.GLAxisItem()
        self.scene.addItem(axis)

        # build meshes for all cameras
        self.meshes = {}
        for port, cam in self.camera_array.cameras.items():
            print(port)
            print(cam)
            mesh = mesh_from_camera(cam)
            self.meshes[port] = mesh
            self.scene.addItem(mesh)

        self.scene.show()

        if self.point_estimates is not None:
            self.scatter = gl.GLScatterPlotItem(
                pos=np.array([0, 0, 0]),
                color = [1,1,1,1],
                size=0.01,
                pxMode=False,
            )
            self.scene.addItem(self.scatter)
        
            self.thread = Thread(target=self.play_data, args=[], daemon=False)
            self.thread.start()

    def play_data(self):
        # sync_indices = self.point_estimate_data["sync_index"].unique().tolist()
        sync_indices = np.unique(self.point_estimates.sync_indices)
        sync_indices = np.sort(sync_indices)

        for sync_index in sync_indices:
            self.display_points(sync_index)
            print(f"Displaying frames from index: {sync_index}")
            time.sleep(1 / 5)

    def display_points(self, sync_index):
        current_sync_index_flag = self.point_estimates.sync_indices == sync_index
        single_board_indices = np.unique(self.point_estimates.obj_indices[current_sync_index_flag])
        
        single_board_points = self.point_estimates.obj[single_board_indices]

        self.scatter.setData(pos=single_board_points)
        # point_data = self.point_estimate_data.query(f"sync_index == {sync_index}")

        # for pair in self.pairs:
            # single_board = point_data.query(f"pair == '{str(pair)}'")
            # x = single_board.x_pos.to_numpy()
            # y = single_board.y_pos.to_numpy()
            # z = single_board.z_pos.to_numpy()

            # board_xyz_pos = np.stack([x, y, z], axis=1)
            # self.scatters[pair].setData(pos=board_xyz_pos)

    def add_point_q(self, q):
        self.point_in_q = q

        board_data = self.point_in_q.get()

        self.color = (1, 0, 0, 1)
        self.board_viz = gl.GLScatterPlotItem(
            pos=board_data.xyz, color=self.color, size=0.01, pxMode=False
        )

        self.scene.addItem(self.board_viz)

    def next_frame(self):
        board_data = self.point_in_q.get()
        self.board_viz.setData(pos=board_data.xyz, color=self.color)

    def begin(self):
        def timer_wrkr():
            while True:
                time.sleep(1 / 15)
                self.next_frame()

        self.timer_thread = Thread(target=timer_wrkr, args=[], daemon=False)
        self.timer_thread.start()


# helper functions to assist with scene creation
def mesh_from_camera(camera_data: CameraData):
    """"
    Mesh is placed at origin by default. Note that it appears rotations
    are in the mesh frame of reference and translations are in 
    the scene frame of reference. I could be wrong, but that appears
    to be the case.
    
    """
    mesh = CameraMesh(camera_data.size, camera_data.matrix).mesh

    # rotate mesh
    euler_angles = rotationMatrixToEulerAngles(camera_data.rotation)
    euler_angles_deg = [x * (180 / math.pi) for x in euler_angles]
    x = euler_angles_deg[0]
    y = euler_angles_deg[1]
    z = euler_angles_deg[2]

    # rotate mesh; z,y,x is apparently the order in which it's done
    # https://gamedev.stackexchange.com/questions/16719/what-is-the-correct-order-to-multiply-scale-rotation-and-translation-matrices-f
    mesh.rotate(z, 0, 0, 1, local=True)
    mesh.rotate(y, 0, 1, 0, local=True)
    mesh.rotate(x, 1, 0, 0, local=True)

    R = camera_data.rotation
    t = camera_data.translation
    
    # adjust mesh translation to account for preliminary rotation
    final_position = t@R.T
    x, y, z = [t for t in final_position]
    mesh.translate(x, y, z)



    return mesh


def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# %%
if __name__ == "__main__":

    from PyQt6.QtWidgets import QApplication

    from pyxyfy import __root__
    from pyxyfy.cameras.camera_array_builder import CameraArrayBuilder
    from pyxyfy.calibration.capture_volume.helper_functions.get_point_estimates import get_point_estimates

    from pyxyfy.calibration.capture_volume.capture_volume import CaptureVolume
    import pickle
    
    # session_directory = Path(__root__,  "tests", "2_cameras_linear")
    session_directory = Path(__root__,  "tests", "tripod")
    # session_directory = Path(__root__,  "tests", "3_cameras_triangular")
    # session_directory = Path(__root__,  "tests", "3_cameras_middle")
    # session_directory = Path(__root__,  "tests", "3_cameras_linear")
    # session_directory = Path(__root__,  "tests", "3_cameras_midlinear")


    saved_CV_path = Path(session_directory, "capture_volume_stage_6.pkl") 
    with open(saved_CV_path, "rb") as f:
        capture_volume:CaptureVolume = pickle.load(f)

    app = QApplication(sys.argv)
    vizr = CaptureVolumeVisualizer(capture_volume = capture_volume)
    # vizr = CaptureVolumeVisualizer(camera_array = capture_volume.camera_array)

    sys.exit(app.exec())