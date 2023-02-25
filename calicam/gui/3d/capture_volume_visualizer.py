import calicam.logger

logger = calicam.logger.get(__name__)

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

from calicam.triangulate.stereo_triangulator import CameraData
from calicam.gui.capture_volume.camera_mesh import CameraMesh
from calicam.cameras.camera_array_builder import CameraArray, CameraArrayBuilder
from calicam.calibration.capture_volume.capture_volume import CaptureVolume

class CaptureVolumeVisualizer:
    def __init__(self, capture_volume:CaptureVolume):
        self.capture_volume = capture_volume
        self.camera_array = capture_volume.camera_array

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

        # read in contents of file and get important parameters
        self.point_history = self.capture_volume.point_history
        # self.pairs = self.point_estimate_data["pair"].unique().tolist()

        # build the initial scatters that will be updated
        # self.scatters = {}
        # self.colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
        # pair_count = len(self.pairs)
        # for pair in self.pairs:
        #     if pair_count == 1:
        #         color = [1,1,1,1]
        #     else:
        #         color = [random(), random(), random(),1]

        #     board_scatter = gl.GLScatterPlotItem(
        #         pos=np.array([0, 0, 0]),
        #         color = color,
        #         size=0.01,
        #         pxMode=False,
        #     )
        #     self.scene.addItem(board_scatter)
        #     self.scatters[pair] = board_scatter

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
        sync_indices = np.unique(self.point_history.sync_indices)
        sync_indices = np.sort(sync_indices)

        for sync_index in sync_indices:
            self.display_points(sync_index)
            print(f"Displaying frames from index: {sync_index}")
            time.sleep(1 / 5)

    def display_points(self, sync_index):
        current_sync_index_flag = self.point_history.sync_indices == sync_index
        single_board_indices = np.unique(self.point_history.obj_indices[current_sync_index_flag])
        
        single_board_points = self.point_history.obj[single_board_indices]

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
def mesh_from_camera(cd: CameraData):
    # cd = camera_data
    mesh = CameraMesh(cd.resolution, cd.camera_matrix).mesh

    # rotate mesh
    logger.info(f"Rotating: {cd.rotation}")
    euler_angles = rotationMatrixToEulerAngles(cd.rotation)
    euler_angles_deg = [x * (180 / math.pi) for x in euler_angles]
    x = euler_angles_deg[0]
    y = euler_angles_deg[1]
    z = euler_angles_deg[2]

    logger.info(f"x: {x}, y: {y}, z: {z}")

    mesh.rotate(x, 1, 0, 0, local=True)
    mesh.rotate(y, 0, 1, 0, local=True)
    mesh.rotate(z, 0, 0, 1, local=True)

    # translate mesh which defaults to origin
    translation_scale_factor = 1
    x, y, z = [t / translation_scale_factor for t in cd.translation]
    mesh.translate(x, y, z)
    logger.info(f"Translation: x: {x}, y: {y}, z: {z}")

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



    from calicam import __root__
    from calicam.cameras.camera_array_builder import CameraArrayBuilder
    from calicam.calibration.capture_volume.point_estimates import (
        get_point_history,
    )
    from calicam.calibration.capture_volume.capture_volume import CaptureVolume
    import pickle
    
    session_directory = Path(__root__, "tests", "5_cameras")
    # stereo_points_csv_path = Path(
        # session_directory, "recording", "stereotriangulated_points.csv"
    # )

    # point_estimate_data = get_point_estimate_data(stereo_points_csv_path)

    # config_path = Path(session_directory, "config.toml")
    # array_builder = CameraArrayBuilder(config_path)
    # camera_array = array_builder.get_camera_array()

    print(f"Optimizing initial camera array configuration ")

    # capture_volume = CaptureVolume(camera_array, point_estimate_data)

    # saved_CV_path = Path(session_directory, "recording", "pre_optimized_capture_volume.pkl") 
    saved_CV_path = Path(session_directory, "recording", "post_optimized_capture_volume.pkl") 
    with open(saved_CV_path, "rb") as f:
        capture_volume = pickle.load(f)
    


    app = QApplication(sys.argv)
    vizr = CaptureVolumeVisualizer(capture_volume)

    sys.exit(app.exec())