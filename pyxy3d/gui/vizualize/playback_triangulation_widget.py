from pathlib import Path
import numpy as np
from time import time
import pyqtgraph.opengl as gl

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyxy3d.gui.vizualize.camera_mesh import CameraMesh, mesh_from_camera
from pyxy3d.cameras.camera_array import CameraArray
import pyxy3d.logger

from pyxy3d.motion_trial import MotionTrial

logger = pyxy3d.logger.get(__name__)
# as part of development process I'm just going to import the skeleton in here

class PlaybackTriangulationWidget(QWidget):
    def __init__(self, camera_array: CameraArray, xyz_history_path: Path = None):
        super(PlaybackTriangulationWidget, self).__init__()

        self.camera_array = camera_array
        self.visualizer = TriangulationVisualizer(self.camera_array)
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self.setMinimumSize(500, 500)

        self.place_widgets()
        self.connect_widgets()
        # going to build out a dictionary of XYZPackets that will be the motion trial
        # TODO: make standalone function def get_motion_trial(xyz_history_path) --> MotionTrial
        # where MotionTrial will be defined in the packets
        self.update_motion_trial(xyz_history_path)

    def place_widgets(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.visualizer.scene)
        self.layout().addWidget(self.slider)

    def connect_widgets(self):
        self.slider.valueChanged.connect(self.visualizer.display_points)
        self.slider.valueChanged.connect(self.visualizer.display_lines)

    def update_motion_trial(self, xyz_history_path):
        # self.xyz_history = pd.read_csv(xyz_history)
        tic = time()
        logger.info(f"Beginning to load in motion trial: {time()}")
        self.motion_trial = MotionTrial(xyz_history_path)
        logger.info(f"Motion trial loading complete: {time()} ")
        toc = time()
        logger.info(f"Elapsed time to load: {toc-tic}")

        self.visualizer.update_motion_trial(self.motion_trial)

        if self.motion_trial.is_empty:
            self.slider.setMinimum(0)
            self.slider.setMaximum(100)
        else:
            self.slider.setMinimum(self.motion_trial.start_index)
            self.slider.setMaximum(self.motion_trial.end_index)

    def update_camera_array(self, camera_array: CameraArray):
        self.visualizer.update_camera_array(camera_array)


class TriangulationVisualizer:

    def __init__(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.build_scene()

    def build_scene(self):
        # constuct a scene if not yet there
        if hasattr(self, "scene"):
            logger.info("Clearing scene in capture volume visualizer")
            self.scene.clear()
        else:
            logger.info("Creating initial scene in capture volume visualizer")
            self.scene = gl.GLViewWidget()

            # the scene camera, not a real Camera
            self.scene.setCameraPosition( distance=4)  
        axis = gl.GLAxisItem()
        self.scene.addItem(axis)

        if self.camera_array.all_extrinsics_calibrated():
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
        
            # will hold a list
        self.segments = {}
        
        self.scene.addItem(self.scatter)
        self.scatter.setData(pos=None)

    def update_camera_array(self, camera_array: CameraArray):
        self.camera_array = camera_array
        self.build_scene()

    def update_motion_trial(self, motion_trial:MotionTrial):
        logger.info("Updating xyz history in playback widget")
        self.motion_trial = motion_trial

        if self.motion_trial.is_empty:
            self.xyz_coord = None
            self.sync_index = 0
            self.segments = None
            self.segment_lines = None
        else:
            self.sync_index = self.motion_trial.start_index

            # x_coord = self.xyz_history["x_coord"]
            # y_coord = self.xyz_history["y_coord"]
            # z_coord = self.xyz_history["z_coord"]
            # self.xyz_coord = np.vstack([x_coord, y_coord, z_coord]).T

            # self.point_ids = self.xyz_history["point_id"]
            # self.segment_lines = {} 

            # for segment in self.segments:
            #     line = gl.GLLinePlotItem(color = pg.mkColor('r'), width= 1, mode="lines" )
            #     self.scene.addItem(line)
            #     self.segment_lines[segment] = line
            # self.scatter.setData(pos=None)

        self.display_points(self.sync_index)

    def display_points(self, sync_index:int):
        """
        sync_index is provided from the dialog and linked to the slider
        it is initially set to the minimum viable sync index
        """

        if self.motion_trial.is_empty:
            self.scatter.setData(pos=None)
        else:
            self.sync_index = sync_index
            logger.debug(f"Displaying xyz points for sync index {sync_index}")
            xyz_coords = self.motion_trial.get_xyz(self.sync_index).point_xyz
            self.scatter.setData(pos=xyz_coords)


    def display_lines(self,sync_index:int):
        if self.segment_lines is not None: 
            self.sync_index = sync_index
            current_sync_index_flag = self.sync_indices == self.sync_index
            current_point_ids = self.point_ids[current_sync_index_flag]
            current_point_xyz = self.xyz_coord[current_sync_index_flag]

            for segment, line in self.segment_lines.items():
                point_id_A = self.segments[segment][0]
                point_id_B = self.segments[segment][1]
                
                xyz_A = current_point_xyz[current_point_ids==point_id_A]
                xyz_B = current_point_xyz[current_point_ids==point_id_B]
                segment_ends = np.vstack([xyz_A,xyz_B])
                line.setData(pos = segment_ends)        
        