#%%
import logging

logging.basicConfig(filename="log\session.log", filemode="w", level=logging.DEBUG)
# level=logging.INFO)
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os.path import exists
from pathlib import Path

import numpy as np
import toml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.charuco import Charuco
from src.cameras.camera import Camera
from src.cameras.video_stream import VideoStream

#%%
MAX_CAMERA_PORT_CHECK = 10


class Session:
    def __init__(self, directory):

        self.dir = str(directory)
        self.config_path = str(Path(self.dir, "config.toml"))

        # dictionary of Cameras, key = port
        self.cameras = {}
        self.streams = {}

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

            # TOML doesn't seem to store None when dumping to file; adjust here
            if "square_size_overide" in self.config["charuco"]:
                sso = self.config["charuco"]["square_size_overide"]
            else:
                sso = None

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

    def load_cameras(self):
        def add_preconfigured_cam(params):
            try:
                port = params["port"]

                self.cameras[port] = Camera(port)

                cam = self.cameras[port]  # trying to make a little more readable
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

    def load_streams(self):
        # need Stream to adjust resolution

        for port, cam in self.cameras.items():
            if port in self.streams.keys():
                pass  # only add if not added yet
            else:
                logging.info(f"Loading Stream for port {port}")
                self.streams[port] = VideoStream(cam)
                # self.stream[port].assign_charuco(self.charuco)

    def adjust_resolutions(self):
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


#%%
if __name__ == "__main__":
    repo = Path(__file__).parent.parent
    config_path = Path(repo, "default_session")
    print(config_path)
    session = Session(config_path)
    session.update_config()
