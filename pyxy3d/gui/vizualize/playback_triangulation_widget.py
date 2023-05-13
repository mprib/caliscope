import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import pyqtgraph.opengl as gl

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyxy3d.session import Session
from pyxy3d.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from pyxy3d.cameras.camera_array import CameraArray


class PlaybackTriangulationWidget(QWidget):
    def __init__(self, camera_array: CameraArray, xyz_history_path: Path = None):
        super(PlaybackTriangulationWidget, self).__init__()

        self.camera_array = camera_array

        self.visualizer = TriangulationVisualizer(self.camera_array)
        # self.visualizer.scene.show()
        self.slider = QSlider(Qt.Orientation.Horizontal)

        # these defaults mean nothing right now without xyz data. Just placeholders

        self.setMinimumSize(500, 500)

        self.place_widgets()
        self.connect_widgets()
        if xyz_history_path is not None:
            xyz_history = pd.read_csv(xyz_history_path)
            self.set_xyz(xyz_history)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)

    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)

    def set_xyz(self, xyz: pd.DataFrame):
        # self.xyz_history = pd.read_csv(xyz_history)
        self.visualizer.set_xyz(xyz)
        if xyz is not None:
            self.slider.setMinimum(self.visualizer.min_sync_index)
            self.slider.setMaximum(self.visualizer.max_sync_index)
        else:
            self.slider.setMinimum(0)
            self.slider.setMaximum(100)


class TriangulationVisualizer:
    """
    Can except either a single camera array or a capture volume that includes
    point_estimates. If a capture volume is supplied, point positions can
    be played back.
    """

    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array

        # constuct a scene
        self.scene = gl.GLViewWidget()
        self.scene.setCameraPosition(distance=4)  # the scene camera, not a real Camera

        axis = gl.GLAxisItem()
        self.scene.addItem(axis)

        # build meshes for all cameras
        self.meshes = {}
        for port, cam in self.camera_array.cameras.items():
            mesh: CameraMesh = mesh_from_camera(cam)
            self.meshes[port] = mesh
            self.scene.addItem(mesh)

            self.scatter = gl.GLScatterPlotItem(
                pos=np.array([0, 0, 0]),
                color=[1, 1, 1, 1],
                size=0.01,
                pxMode=False,
            )
            self.scene.addItem(self.scatter)
            self.scatter.setData(pos=None)

    def set_xyz(self, xyz_history: pd.DataFrame):
        self.xyz_history = xyz_history

        if self.xyz_history is not None:
            self.sync_indices = self.xyz_history["sync_index"]
            self.min_sync_index = np.min(self.sync_indices)
            self.max_sync_index = np.max(self.sync_indices)
            self.sync_index = self.min_sync_index

            x_coord = self.xyz_history["x_coord"]
            y_coord = self.xyz_history["y_coord"]
            z_coord = self.xyz_history["z_coord"]
            self.xyz_coord = np.vstack([x_coord, y_coord, z_coord]).T

        else:
            self.xyz_coord = None
            # self.scatter.setData(pos=None)

        self.display_points(self.sync_index)

    def display_points(self, sync_index):
        """
        sync_index is provided from the dialog and linked to the slider
        it is initially set to the minimum viable sync index
        """
        self.sync_index = sync_index

        current_sync_index_flag = self.sync_indices == self.sync_index

        if self.xyz_coord is not None:
            self.points = self.xyz_coord[current_sync_index_flag]
            logger.info(f"Displaying xyz points for sync index {sync_index}")
            self.scatter.setData(pos=self.points)

        else: 
            self.scatter.setData(pos=None)