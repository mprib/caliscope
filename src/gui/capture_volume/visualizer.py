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

import numpy as np
import pyqtgraph.opengl as gl

from src.triangulate.stereo_triangulator import CameraData
from src.gui.capture_volume.camera_mesh import CameraMesh


class CaptureVolumeVisualizer:
    def __init__(self, triangulator):
        self.triangulator = triangulator
        self.point_in_q = None
        self.mesh_A = mesh_from_camera(triangulator.camera_A)
        self.mesh_B = mesh_from_camera(triangulator.camera_B)

        # create the overhead for display
        # self.app = pg.mkQApp("Stereo Visualizer")
        self.scene = gl.GLViewWidget()
        # self.scene.setWindowTitle("Camera Calibration")
        self.scene.setCameraPosition(distance=4)

        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)

        self.scene.addItem(grid)
        self.scene.addItem(self.mesh_A)
        self.scene.addItem(self.mesh_B)

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

    from PyQt6.QtWidgets import QApplication

    from src.calibration.charuco import Charuco
    from src.calibration.corner_tracker import CornerTracker
    from src.cameras.synchronizer import Synchronizer
    from src.recording.recorded_stream import RecordedStreamPool
    from src.triangulate.paired_point_stream import PairedPointStream
    from src.triangulate.stereo_triangulator import StereoTriangulator
    from src.gui.capture_volume.visualizer import CaptureVolumeVisualizer
    from src.cameras.camera_array import CameraArray, CameraArrayBuilder

    # set the location for the sample data used for testing
    repo = str(Path(__file__)).split("src")[0]

    # session_directory = Path(repo, "sessions", "high_res_session")
    calibration_data = Path(repo, "sessions", "iterative_adjustment")
    video_directory = Path(calibration_data, "recording")
    # create playback streams to provide to synchronizer
    ports = [0, 2]

    camera_array = CameraArrayBuilder(calibration_data).get_camera_array()

    recorded_stream_pool = RecordedStreamPool(ports, video_directory)
    syncr = Synchronizer(recorded_stream_pool.streams, fps_target=None)
    recorded_stream_pool.play_videos()
    # create a corner tracker to locate board corners
    charuco = Charuco(
        4, 5, 11, 8.5, aruco_scale=0.75, square_size_overide_cm=5.25, inverted=True
    )
    trackr = CornerTracker(charuco)

    # create a commmon point finder to grab charuco corners shared between the pair of ports
    pairs = [(ports[0], ports[1])]
    point_stream = PairedPointStream(
        synchronizer=syncr,
        pairs=pairs,
        tracker=trackr,
    )

    camA, camB = camera_array.cameras[0], camera_array.cameras[2]
    pair = (camA.port, camB.port)

    config_path = str(Path(calibration_data, "config.toml"))
    test_pair_in_q = Queue(-1)

    triangulatr = StereoTriangulator(camA, camB, test_pair_in_q)

    # create a thread that will feed the test_pair_in_q with new datapoints
    def coordinate_feeder_worker():
        while True:
            point_packet = point_stream.out_q.get()
            if point_packet.pair == pairs[0]:
                test_pair_in_q.put(point_packet)

    thread = Thread(target=coordinate_feeder_worker, args=[], daemon=False)
    thread.start()

    app = QApplication(sys.argv)
    vizr = CaptureVolumeVisualizer(triangulatr)
    vizr.add_point_q(triangulatr.out_q)
    vizr.scene.show()
    vizr.begin()

    sys.exit(app.exec())
