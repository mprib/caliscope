import logging
import pprint

LOG_FILE = r"log\stereo_visualizer.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
import math
import numpy as np
from pathlib import Path
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.triangulate.visualization.camera_mesh import CameraMesh
from src.triangulate.stereo_triangulator import CameraData


class StereoVisualizer:
    def __init__(self, triangulator):
        self.triangulator = triangulator

        self.mesh_A = mesh_from_camera(triangulator.camera_A)
        self.mesh_B = mesh_from_camera(triangulator.camera_B)

        # create the overhead for display
        self.app = pg.mkQApp("Stereo Visualizer")
        self.scene = gl.GLViewWidget()
        self.scene.setWindowTitle("Camera Calibration")
        self.scene.setCameraPosition(distance=4)

        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)

        self.scene.addItem(grid)
        self.scene.addItem(self.mesh_A)
        self.scene.addItem(self.mesh_B)

        self.scene.show()
        # self.add_test_scatter()

    def add_test_scatter(self):

        self.phase = 0  # working variable while figuring out scatter
        ##### START NEW TEST PYQTGRAPH STUFF
        self.pos3 = np.zeros((10, 10, 3))
        self.pos3[:, :, :2] = np.mgrid[:10, :10].transpose(1, 2, 0) * [-0.1, 0.1]
        self.pos3 = self.pos3.reshape(100, 3)
        self.d3 = (self.pos3**2).sum(axis=1) ** 0.5
        self.color = (1, 1, 1, 0.1)
        self.sp3 = gl.GLScatterPlotItem(
            pos=self.pos3, color=self.color, size=0.1, pxMode= False
        )

        self.scene.addItem(self.sp3)

    def add_test_board_scatter(self, board_data):
        self.board_data = board_data   

        self.color = (1, 0, 0, 1)
        self.sp1 = gl.GLScatterPlotItem(
            pos=self.board_data , color=self.color, size=.1, pxMode=False
        )

        self.scene.addItem(self.sp1)


    def update(self):
        # update surface positions and colors
        # update volume colors
        self.phase -= 0.1
        z = -np.cos(self.d3*2+self.phase)
        self.pos3[:,2] = z

        self.sp3.setData(pos=self.pos3, color=self.color)
        ################ END TEST NEW PYQTGRAPHSTUFF

    def start(self):
        t = QtCore.QTimer()
        t.timeout.connect(self.update)
        t.start(50)
        pg.exec()


# helper functions to assist with scene creation
def mesh_from_camera(cd: CameraData):
    # cd = camera_data
    mesh = CameraMesh(cd.resolution, cd.camera_matrix).mesh

    # translate mesh which defaults to origin
    translation_scale_factor = 1
    x, y, z = [t / translation_scale_factor for t in cd.translation]
    mesh.translate(x, y, z)
    logging.info(f"Translation: x: {x}, y: {y}, z: {z}")

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


if __name__ == "__main__":

    from src.triangulate.stereo_triangulator import StereoTriangulator

    from src.recording.recorded_stream import RecordedStreamPool
    from src.cameras.synchronizer import Synchronizer
    from src.calibration.charuco import Charuco
    from src.triangulate.paired_point_stream import PairedPointStream
    from src.triangulate.stereo_triangulator import StereoTriangulator
    from src.triangulate.visualization.stereo_visualizer import StereoVisualizer
    from src.calibration.corner_tracker import CornerTracker

    # set the location for the sample data used for testing
    repo = Path(__file__).parent.parent.parent.parent
    session_directory =Path(repo, "examples", "high_res_session")
    # create playback streams to provide to synchronizer
    ports = [0, 1]
    recorded_stream_pool = RecordedStreamPool(ports, session_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()
    # create a corner tracker to locate board corners
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide=0.0525, inverted=True
    )
    trackr = CornerTracker(charuco)
    # create a commmon point finder to grab charuco corners shared between the pair of ports
    pairs = [(0, 1)]
    point_stream = PairedPointStream(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,
    )
    config_path = str(Path(session_directory, "config.toml"))
    triangulatr = StereoTriangulator(point_stream, config_path)

    test_board_corners = np.array(
        [
            [-8.97169043, 8.58954463, 56.44145603],
            [-9.28875076, 4.38558929, 54.45100235],
            [-9.65045165, 0.20856289, 52.33586731],
            [-13.50551906, 9.2943572, 55.70972394],
            [-13.80153222, 5.0210905, 53.37978951],
            [-14.07910444, 0.84995609, 51.34233615],
            [-17.93996941, 9.86696736, 54.15030867],
            [-18.18285319, 5.69386964, 52.21569521],
            [-18.5111405, 1.50492472, 49.85212552],
            [-22.37649957, 10.55513425, 53.03158067],
            [-22.6366339, 6.28469088, 50.85093531],
            [-22.8193701, 2.1501825, 48.45748189],
        ]
    )

    test_board_corners = test_board_corners/10
    vizr = StereoVisualizer(triangulatr)
    vizr.add_test_scatter()
    vizr.add_test_board_scatter(test_board_corners)
    vizr.start()
