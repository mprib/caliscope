# Manager of all created objects and the primary interface for the GUI.


import logging

LOG_FILE = "log\session.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO

LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os.path import exists
from pathlib import Path

import numpy as np
import toml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.charuco import Charuco
from src.calibration.corner_tracker import CornerTracker
from src.calibration.monocalibrator import MonoCalibrator
from src.calibration.stereocalibrator import StereoCalibrator
from src.cameras.camera import Camera
from src.cameras.synchronizer import Synchronizer
from src.cameras.live_stream import LiveStream
from src.gui.stereo_frame_builder import StereoFrameBuilder

#%%
MAX_CAMERA_PORT_CHECK = 10


class Session:
    def __init__(self, directory):

        self.dir = str(directory)
        self.config_path = str(Path(self.dir, "config.toml"))

        # dictionaries of streaming related objects. key = port
        self.cameras = {}
        self.streams = {}

        # dictionaries of calibration related objects.
        self.monocalibrators = {}  # key = port

        self.load_config()
        self.load_charuco()

    def load_config(self):

        if exists(self.config_path):
            logging.info("Found previous config")
            with open(self.config_path, "r") as f:
                self.config = toml.load(self.config_path)
        else:
            logging.info("Creating it")

            self.config = toml.loads("")
            self.config["CreationDate"] = datetime.now()
            with open(self.config_path, "a") as f:
                toml.dump(self.config, f)

        return self.config

    def update_config(self):

        # alphabetize by key
        sorted_config = {key: value for key, value in sorted(self.config.items())}
        self.config = sorted_config

        with open(self.config_path, "w") as f:
            toml.dump(self.config, f)

    def load_charuco(self):

        if "charuco" in self.config:
            logging.info("Loading charuco from config")
            params = self.config["charuco"]

            self.charuco = Charuco(
                columns=params["columns"],
                rows=params["rows"],
                board_height=params["board_height"],
                board_width=params["board_width"],
                dictionary=params["dictionary"],
                units=params["units"],
                aruco_scale=params["aruco_scale"],
                square_size_overide=params["square_size_overide"],
                inverted=params["inverted"],
            )
        else:
            logging.info("Loading default charuco")
            self.charuco = Charuco(4, 5, 11, 8.5, square_size_overide=5.4)
            self.config["charuco"] = self.charuco.__dict__
            self.update_config()

    def save_charuco(self):
        self.config["charuco"] = self.charuco.__dict__
        logging.info(f"Saving charuco with params {self.charuco.__dict__} to config")
        self.update_config()
        
    def delete_camera(self, port_to_delete):
        for key, params in self.config.copy().items():
            if key.startswith("cam"):
                port = params["port"]
                if port == port_to_delete:
                    del self.config[key]
                    
    def delete_all_cam_data(self):
        for key, params in self.config.copy().items():
            if key.startswith("cam"):
                del self.config[key]
            if key.startswith("stereo"):
                del self.config[key]
                
        self.update_config()

    def load_cameras(self):
        def add_preconfigured_cam(params):
            try:
                port = params["port"]
                self.cameras[port] = Camera(port)

                cam = self.cameras[port]
                cam.rotation_count = params["rotation_count"]
                cam.exposure = params["exposure"]
            except:
                logging.info("Unable to connect... camera may be in use.")

            # if calibration done, then populate those
            if "error" in params.keys():
                logging.info(params["error"])
                cam.error = params["error"]
                cam.camera_matrix = np.array(params["camera_matrix"]).astype(float)
                cam.distortion = np.array(params["distortion"]).astype(float)
                cam.grid_count = params["grid_count"]

        with ThreadPoolExecutor() as executor:
            for key, params in self.config.items():
                if key.startswith("cam"):
                    if params["port"] in self.streams.keys():
                        logging.info(f"Don't reload a camera at port {params['port']}")
                    else:
                        logging.info(f"Beginning to load {key} with params {params}")
                        executor.submit(add_preconfigured_cam, params)

    def find_additional_cameras(self):
        def add_cam(port):
            try:
                logging.info(f"Trying port {port}")
                cam = Camera(port)
                logging.info(f"Success at port {port}")
                self.cameras[port] = cam
                self.save_camera(port)
            except:
                logging.info(f"No camera at port {port}")

        with ThreadPoolExecutor() as executor:
            for i in range(0, MAX_CAMERA_PORT_CHECK):
                if i in self.cameras.keys():
                    # don't try to connect to an already connected camera
                    pass
                else:
                    executor.submit(add_cam, i)

    def load_stream_tools(self):
        # in addition to populating the active streams, this loads a frame synchronizer

        for port, cam in self.cameras.items():
            if port in self.streams.keys():
                pass  # only add if not added yet
            else:
                logging.info(f"Loading Stream for port {port}")
                self.streams[port] = LiveStream(cam)

        self.synchronizer = Synchronizer(self.streams, fps_target=6.2)

    def load_monocalibrators(self):
        self.corner_tracker = CornerTracker(self.charuco)

        for port, cam in self.cameras.items():
            if port in self.monocalibrators.keys():
                pass  # only add if not added yet
            else:
                logging.info(f"Loading Monocalibrator for port {port}")
                self.monocalibrators[port] = MonoCalibrator(
                    cam, self.synchronizer, self.corner_tracker
                )

    def load_stereo_tools(self):
        self.corner_tracker = CornerTracker(self.charuco)
        self.stereocalibrator = StereoCalibrator(self.synchronizer, self.corner_tracker)
        self.stereo_frame_builder = StereoFrameBuilder(self.stereocalibrator)

    def adjust_resolutions(self):
        """Changes the camera resolution to the value in the configuration, as
        log as it is not configured for the default resolution"""

        def adjust_res_worker(port):
            stream = self.streams[port]
            resolution = self.config[f"cam_{port}"]["resolution"]
            default_res = self.cameras[port].default_resolution
            logging.info(f"Port {port} resolution is {resolution[0:2]}")
            logging.info(f"Port {port} default res is {default_res[0:2]}")

            if resolution[0] != default_res[0] or resolution[1] != default_res[1]:
                logging.info(f"Attempting to change resolution on port {port}")
                stream.change_resolution(resolution)

        with ThreadPoolExecutor() as executor:
            for port in self.cameras.keys():
                executor.submit(adjust_res_worker, port)

    def save_camera(self, port):
        cam = self.cameras[port]
        params = {
            "port": cam.port,
            "resolution": cam.resolution,
            "rotation_count": cam.rotation_count,
            "error": cam.error,
            "camera_matrix": cam.camera_matrix,
            "distortion": cam.distortion,
            "exposure": cam.exposure,
            "grid_count": cam.grid_count,
        }

        logging.info(f"Saving camera parameters...{params}")

        self.config["cam_" + str(port)] = params
        self.update_config()

    def save_stereocalibration(self):
        logging.info(f"Saving stereocalibration....")
        logging.info(self.stereocalibrator.stereo_outputs)

        stereo_out = self.stereocalibrator.stereo_outputs
        for pair, stereo_params in stereo_out.items():
            config_key = f"stereo_{pair[0]}_{pair[1]}"
            self.config[config_key] = stereo_params

        self.update_config()


#%%
if __name__ == "__main__":
    repo = Path(__file__).parent.parent
    config_path = Path(repo, "examples", "default_session")
    print(config_path)
    session = Session(config_path)
    session.update_config()
