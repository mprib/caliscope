import logging

LOG_FILE = r"log\stereo_visualizer.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import math
import sys
import time
from pathlib import Path
from threading import Thread
from queue import Queue
import pandas as pd
import numpy as np
import pyqtgraph.opengl as gl

from calicam.triangulate.stereo_triangulator import CameraData
from calicam.gui.capture_volume.camera_mesh import CameraMesh
from calicam.cameras.camera_array import CameraArray, CameraArrayBuilder


class CaptureVolumeVisualizer:
    def __init__(self, camera_array: CameraArray, xyz_pos_path: Path = None):

        self.camera_array = camera_array

        self.current_frame = 0

        # constuct a scene
        self.scene = gl.GLViewWidget()
        self.scene.setCameraPosition(distance=4)



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
        if xyz_pos_path is not None:

            # read in contents of file and get important parameters
            self.point_data = pd.read_csv(xyz_pos_path)
            self.pairs = self.point_data["pair"].unique().tolist()

            # build the initial scatters that will be updated
            self.scatters = {}
            self.colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
            for pair in self.pairs:

                board_scatter = gl.GLScatterPlotItem(
                    pos=np.array([0, 0, 0]),
                    color=self.colors.pop(),
                    size=0.01,
                    pxMode=False,
                )
                self.scene.addItem(board_scatter)
                self.scatters[pair] = board_scatter
   
            self.thread = Thread(target=self.play_data, args=[], daemon=False)
            self.thread.start()

    def play_data(self):
            sync_indices = self.point_data["sync_index"].unique().tolist()
            for sync_index in sync_indices:
                self.display_points(sync_index)
                print(f"Displaying frames from index: {sync_index}")
                time.sleep(1/10)
    
    def display_points(self, sync_index):

        point_data = self.point_data.query(f"sync_index == {sync_index}")

        for pair in self.pairs:
            single_board = point_data.query(f"pair == '{str(pair)}'")
            x = single_board.x_pos.to_numpy()
            y = single_board.y_pos.to_numpy()
            z = single_board.z_pos.to_numpy()

            board_xyz_pos = np.stack([x, y, z], axis=1)
            self.scatters[pair].setData(pos=board_xyz_pos)

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
        print(board_data.time)
        self.board_viz.setData(pos=board_data.xyz, color=self.color)

    def begin(self):
        def timer_wrkr():
            while True:
                time.sleep(1 / 30)
                self.next_frame()

        self.timer_thread = Thread(target=timer_wrkr, args=[], daemon=False)
        self.timer_thread.start()


# helper functions to assist with scene creation
def mesh_from_camera(cd: CameraData):
    # cd = camera_data
    mesh = CameraMesh(cd.resolution, cd.camera_matrix).mesh

    # rotate mesh
    logging.info(f"Rotating: {cd.rotation}")
    euler_angles = rotationMatrixToEulerAngles(cd.rotation)
    euler_angles_deg = [x * (180 / math.pi) for x in euler_angles]
    x = euler_angles_deg[0]
    y = euler_angles_deg[1]
    z = euler_angles_deg[2]

    logging.info(f"x: {x}, y: {y}, z: {z}")

    mesh.rotate(x, 1, 0, 0, local=True)
    mesh.rotate(y, 0, 1, 0, local=True)
    mesh.rotate(z, 0, 0, 1, local=True)

    
    # translate mesh which defaults to origin
    translation_scale_factor = 1
    x, y, z = [t / translation_scale_factor for t in cd.translation]
    mesh.translate(x, y, z) 
    logging.info(f"Translation: x: {x}, y: {y}, z: {z}")


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

    # set the location for the sample data used for testing
    repo = str(Path(__file__)).split("src")[0]

    config_path = Path(repo, "sessions", "iterative_adjustment", "config.toml")
    camera_array = CameraArrayBuilder(config_path).get_camera_array()

    point_data_path = Path(
        repo, "sessions", "iterative_adjustment", "recording", "triangulated_points.csv"
        # repo, "sessions", "iterative_adjustment", "recording", "triangulated_points_bundle_adjusted.csv"
        # repo, "sessions", "iterative_adjustment", "recording", "triangulated_points_daisy_chain.csv"
        # repo, "sessions", "iterative_adjustment", "recording", "triangulated_points_bundle_adjusted_300.csv"
        # repo, "sessions", "iterative_adjustment", "recording", "triangulated_points_daisy_chain_300.csv"
    )

    app = QApplication(sys.argv)
    vizr = CaptureVolumeVisualizer(camera_array, point_data_path)

    sys.exit(app.exec())
