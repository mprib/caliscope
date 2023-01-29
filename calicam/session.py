# Environment for managing all created objects and the primary interface for the GUI.


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
from pathlib import Path, PurePath
from enum import Enum, auto
import numpy as np
import toml
from itertools import combinations

from calicam.calibration.charuco import Charuco
from calicam.calibration.corner_tracker import CornerTracker
from calicam.calibration.monocalibrator import MonoCalibrator
from calicam.calibration.stereocalibrator import StereoCalibrator
from calicam.cameras.camera import Camera
from calicam.cameras.synchronizer import Synchronizer
from calicam.cameras.live_stream import LiveStream
from calicam.recording.video_recorder import VideoRecorder
from calicam.gui.stereo_calibration.stereo_frame_builder import StereoFrameBuilder
from calicam.gui.stereo_calibration.stereo_frame_emitter import StereoFrameEmitter

#%%
MAX_CAMERA_PORT_CHECK = 10


class Session:
    def __init__(self, directory):

        self.folder = PurePath(directory).name
        self.path = directory
        self.config_path = str(Path(directory, "config.toml"))

        # dictionaries of streaming related objects. key = port
        self.cameras = {}
        self.streams = {}

        # dictionaries of calibration related objects.
        self.monocalibrators = {}  # key = port

        self.synchronizer_created = False

        self.load_config()
        self.load_charuco()

    def load_config(self):

        if exists(self.config_path):
            logging.info("Found previous config")
            with open(self.config_path, "r") as f:
                self.config = toml.load(self.config_path)
        else:
            logging.info(
                "No existing config.toml found; creating starter file with charuco"
            )

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
                square_size_overide_cm=params["square_size_overide_cm"],
                inverted=params["inverted"],
            )
        else:
            logging.info("Loading default charuco")
            self.charuco = Charuco(4, 5, 11, 8.5, square_size_overide_cm=5.4)
            self.config["charuco"] = self.charuco.__dict__
            self.update_config()

    def save_charuco(self):
        self.config["charuco"] = self.charuco.__dict__
        logging.info(f"Saving charuco with params {self.charuco.__dict__} to config")
        self.update_config()

    def delete_camera(self, port_to_delete):
        # note: needs to be a copy to avoid errors while dict changes with deletion
        for key, params in self.config.copy().items():
            if key.startswith("cam"):
                port = params["port"]
                if port == port_to_delete:
                    del self.config[key]

    def delete_all_cam_data(self):
        # note: needs to be a copy to avoid errors while dict changes with deletion
        for key, params in self.config.copy().items():
            if key.startswith("cam"):
                del self.config[key]
            if key.startswith("stereo"):
                del self.config[key]

        self.update_config()

    def connected_camera_count(self):
        cam_count = 0

        # for key in self.config.keys():
        #     if key.startswith("cam"):
        #         cam_count+=1
        return len(self.cameras)

    def calibrated_camera_count(self):
        """Used to keep track of where the user is in the calibration process"""
        count = 0
        for key in self.config.keys():
            if key.startswith("cam"):
                if "error" in self.config[key].keys():
                    if self.config[key]["error"] is not None:
                        count += 1
        return count

    def camera_pairs(self):
        """Used to keep track of where the user is in the calibration process"""
        ports = [key for key in self.cameras.keys()]
        pairs = [pair for pair in combinations(ports, 2)]
        sorted_ports = [
            (min(pair), max(pair)) for pair in pairs
        ]  # sort as in (b,a) --> (a,b)
        sorted_ports = sorted(
            sorted_ports
        )  # sort as in [(b,c), (a,b)] --> [(a,b), (b,c)]
        return sorted_ports

    def calibrated_camera_pairs(self):
        """Used to keep track of where the user is in the calibration process"""
        calibrated_pairs = []
        for key in self.config.keys():
            if key.startswith("stereo"):
                portA, portB = key.split("_")[1:3]
                calibrated_pairs.append((int(portA), int(portB)))
        calibrated_pairs = sorted(
            calibrated_pairs
        )  # sort as in [(b,c), (a,b)] --> [(a,b), (b,c)]
        return calibrated_pairs

    def load_cameras(self):

        # worker function that will be spun up to connect to a previously configured camera
        def add_preconfigured_cam(params):
            try:
                port = params["port"]
                logging.info(f"Attempting to add pre-configured camera at port {port}")

                if params["ignore"]:
                    logging.info(f"Ignoring camera at port {port}")
                    pass  # don't load it in
                else:
                    self.cameras[port] = Camera(port)
                    cam = self.cameras[port]  # just for ease of reference
                    cam.rotation_count = params["rotation_count"]
                    cam.exposure = params["exposure"]

                    # if calibration done, then populate those as well
                    if "error" in params.keys():
                        logging.info(
                            f"Camera RMSE error for port {port}: {params['error']}"
                        )
                        cam.error = params["error"]
                        cam.camera_matrix = np.array(params["camera_matrix"]).astype(
                            float
                        )
                        cam.distortion = np.array(params["distortion"]).astype(float)
                        cam.grid_count = params["grid_count"]
            except:
                logging.info("Unable to connect... camera may be in use.")

        with ThreadPoolExecutor() as executor:
            for key, params in self.config.items():
                if key.startswith("cam"):
                    if params["port"] in self.streams.keys():
                        logging.info(f"Don't reload a camera at port {params['port']}")
                    else:
                        logging.info(f"Beginning to load {key} with params {params}")
                        executor.submit(add_preconfigured_cam, params)

    def find_cameras(self):
        """Attempt to connect to the first N cameras. It will clear out any previous calibration
        data, including stereocalibration data"""

        def add_cam(port):
            try:
                logging.info(f"Trying port {port}")
                cam = Camera(port)
                logging.info(f"Success at port {port}")
                self.cameras[port] = cam
                self.save_camera(port)
                self.streams[port] = LiveStream(cam)
            except:
                logging.info(f"No camera at port {port}")

        with ThreadPoolExecutor() as executor:
            for i in range(0, MAX_CAMERA_PORT_CHECK):
                if i in self.cameras.keys():
                    # don't try to connect to an already connected camera
                    pass
                else:
                    executor.submit(add_cam, i)

        # remove potential stereocalibration data

        for key in self.config.copy().keys():
            if key.startswith("stereo"):
                del self.config[key]
        self.update_config()

    def load_streams(self):
        # in addition to populating the active streams, this loads a frame synchronizer

        for port, cam in self.cameras.items():
            if port in self.streams.keys():
                pass  # only add if not added yet
            else:
                logging.info(f"Loading Stream for port {port}")
                self.streams[port] = LiveStream(cam)

    def disconnect_cameras(self):
        """Destroy all camera reading associated threads working down to the cameras
        themselves so that the session cameras can be later reconstructed (potentially
        with additional or fewer cameras)"""

        try:
            logging.info("Attempting to shutdown monocalibrators")
            for port, monocal in self.monocalibrators.items():
                monocal.stop()
                # monocal.thread.join()

            self.monocalibrators = {}
        except (AttributeError):
            logging.warning("No monocalibrators to delete")
            pass

        try:
            logging.info("Attempting to stop stereo frame emitter")
            self.stereo_frame_emitter.stop()
            # self.stereo_frame_emitter.thread.join()

        except (AttributeError):
            logging.info("No stereo frame emitter to stop")

        try:
            logging.info("Attempting to stop stereocalibrator")
            self.stereocalibrator.stop()

        except (AttributeError):
            logging.warning("No stereocalibrator to delete.")
            pass  # don't worry if it doesn't exist

        try:
            logging.info("Attempting to stop synchronizer...")

            self.synchronizer.stop()
            del (
                self.synchronizer
            )  # important for session to know to recreate stereotools
        except (AttributeError):
            logging.warning("No synchronizer to delete")
            pass

        try:
            logging.info("Attempting to stop streams...")
            for port, stream in self.streams.items():
                stream.stop()
            self.streams = {}

            for port, cam in self.cameras.items():
                cam.capture.release()
                logging.info(f"Capture released at port {port}")
                # del cam
            # del self.cameras
            self.cameras = {}
        except (AttributeError):

            logging.warning("Unable to delete all streams...")
            pass

    def load_monocalibrators(self):
        self.corner_tracker = CornerTracker(self.charuco)

        for port, cam in self.cameras.items():
            if port in self.monocalibrators.keys():
                logging.info(
                    f"Skipping over monocalibrator creation for port {port} because it already exists."
                )
                pass  # only add if not added yet
            else:
                logging.info(f"Loading Monocalibrator for port {port}")
                self.monocalibrators[port] = MonoCalibrator(
                    self.streams[port], self.corner_tracker
                )

    def remove_monocalibrators(self):
        for port, monocal in self.monocalibrators.copy().items():
            logging.info(f"Attempting to stop Monocalibrator for port {port}")
            monocal.stop()
            del self.monocalibrators[port]
            logging.info(f"Successfuly stopped monocalibrator at port {port}")

    def load_stereo_tools(self):
        if hasattr(self, "synchronizer"):
            logging.info("No stereotools created...synchronizer already exists")
        else:
            logging.info("Creating stereo tools...")
            self.synchronizer = Synchronizer(self.streams)
            self.corner_tracker = CornerTracker(self.charuco)
            self.stereocalibrator = StereoCalibrator(
                self.synchronizer, self.corner_tracker
            )
            self.stereo_frame_builder = StereoFrameBuilder(self.stereocalibrator)
            self.stereo_frame_emitter = StereoFrameEmitter(self.stereo_frame_builder)
            self.stereo_frame_emitter.start()

    def remove_stereo_tools(self):
        self.stereocalibrator.stop()
        del self.stereocalibrator
        self.synchronizer.stop()
        del self.synchronizer
        # self.stereo_frame_builder
        # self.stereo_frame_emitter

    def load_video_recorder(self):
        if hasattr(self, "synchronizer"):
            self.video_recorder = VideoRecorder(self.synchronizer)
        else:
            logging.warning("No synchronizer available to record video")

    def adjust_resolutions(self):
        """Changes the camera resolution to the value in the configuration, as
        log as it is not configured for the default resolution"""

        def adjust_res_worker(port):
            stream = self.streams[port]
            resolution = self.config[f"cam_{port}"]["resolution"]
            default_res = self.cameras[port].default_resolution

            if resolution[0] != default_res[0] or resolution[1] != default_res[1]:
                logging.info(
                    f"Beginning to change resolution at port {port} from {default_res[0:2]} to {resolution[0:2]}"
                )
                stream.change_resolution(resolution)
                logging.info(
                    f"Completed change of resolution at port {port} from {default_res[0:2]} to {resolution[0:2]}"
                )

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
            "ignore": cam.ignore,
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

    def get_stage(self):
        if self.connected_camera_count() == 0:
            return stage.NO_CAMERAS

        if self.calibrated_camera_count() < self.connected_camera_count():
            return stage.UNCALIBRATED_CAMERAS

        if len(self.calibrated_camera_pairs()) == len(self.camera_pairs()):
            return stage.STEREOCALIBRATION_DONE

        if (
            self.connected_camera_count() > 0
            and self.calibrated_camera_count() == self.connected_camera_count()
        ):
            return stage.MONOCALIBRATED_CAMERAS


class stage(Enum):
    NO_CAMERAS = auto()
    UNCALIBRATED_CAMERAS = auto()
    MONOCALIBRATED_CAMERAS = auto()
    STEREOCALIBRATION_IN_PROCESS = auto()
    STEREOCALIBRATION_DONE = auto()
    ORIGIN_SET = auto()


#%%
if __name__ == "__main__":
    repo = Path(__file__).parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    print(config_path)
    print("Loading session config")
    session = Session(config_path)
    print(session.get_stage())
    session.update_config()
    # print("Loading Cameras...")
    # session.load_cameras()

    print("Finding Cameras...")
    session.find_cameras()
    print(session.get_stage())
    print(f"Camera pairs: {session.camera_pairs()}")
    print(f"Calibrated Camera pairs: {session.calibrated_camera_pairs()}")
    session.disconnect_cameras()
    print(session.get_stage())
    print(f"Camera pairs: {session.camera_pairs()}")
    print(f"Calibrated Camera pairs: {session.calibrated_camera_pairs()}")
