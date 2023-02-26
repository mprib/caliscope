# Environment for managing all created objects and the primary interface for the GUI.
import calicam.logger

logger = calicam.logger.get(__name__)

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
from calicam.cameras.camera import Camera
from calicam.cameras.synchronizer import Synchronizer
from calicam.cameras.live_stream import LiveStream
from calicam.recording.video_recorder import VideoRecorder

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
            logger.info("Found previous config")
            with open(self.config_path, "r") as f:
                self.config = toml.load(self.config_path)
        else:
            logger.info(
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
            logger.info("Loading charuco from config")
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
            logger.info("Loading default charuco")
            self.charuco = Charuco(4, 5, 11, 8.5, square_size_overide_cm=5.4)
            self.config["charuco"] = self.charuco.__dict__
            self.update_config()

    def save_charuco(self):
        self.config["charuco"] = self.charuco.__dict__
        logger.info(f"Saving charuco with params {self.charuco.__dict__} to config")
        self.update_config()

    def delete_camera(self, port_to_delete):
        # note: needs to be a copy to avoid errors while dict changes with deletion
        for key, params in self.config.copy().items():
            if key.startswith("cam"):
                port = params["port"]
                if port == port_to_delete:
                    del self.config[key]

    def get_configured_camera_count(self):
        count = 0
        for key, params in self.config.copy().items():
            if key.startswith("cam"):
                count +=1
        return count
    
    def delete_all_cam_data(self):
        # note: needs to be a copy to avoid errors while dict changes with deletion
        for key, params in self.config.copy().items():
            if key.startswith("cam"):
                del self.config[key]
            if key.startswith("stereo"):
                del self.config[key]

        self.update_config()

    def connected_camera_count(self):
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
                logger.info(f"Attempting to add pre-configured camera at port {port}")

                if params["ignore"]:
                    logger.info(f"Ignoring camera at port {port}")
                    pass  # don't load it in
                else:
                    if "verified_resolutions" in params.keys():
                        verified_resolutions = params["verified_resolutions"]
                        self.cameras[port] = Camera(port, verified_resolutions)
                    else:
                        self.cameras[port] = Camera(port)

                    cam = self.cameras[port]  # just for ease of reference
                    cam.rotation_count = params["rotation_count"]
                    cam.exposure = params["exposure"]

                    # if calibration done, then populate those as well
                    if "error" in params.keys():
                        logger.info(
                            f"Camera RMSE error for port {port}: {params['error']}"
                        )
                        cam.error = params["error"]
                        cam.camera_matrix = np.array(params["camera_matrix"]).astype(
                            float
                        )
                        cam.distortion = np.array(params["distortion"]).astype(float)
                        cam.grid_count = params["grid_count"]
            except:
                logger.info("Unable to connect... camera may be in use.")

        with ThreadPoolExecutor() as executor:
            for key, params in self.config.items():
                if key.startswith("cam"):
                    if params["port"] in self.streams.keys():
                        logger.info(f"Don't reload a camera at port {params['port']}")
                    else:
                        logger.info(f"Beginning to load {key} with params {params}")
                        executor.submit(add_preconfigured_cam, params)

    def find_cameras(self):
        """Attempt to connect to the first N cameras. It will clear out any previous calibration
        data, including stereocalibration data"""

        def add_cam(port):
            try:
                logger.info(f"Trying port {port}")
                cam = Camera(port)
                logger.info(f"Success at port {port}")
                self.cameras[port] = cam
                self.save_camera(port)
                self.streams[port] = LiveStream(cam, charuco = self.charuco)
            except:
                logger.info(f"No camera at port {port}")

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
                logger.info(f"Loading Stream for port {port}")
                self.streams[port] = LiveStream(cam, charuco=self.charuco)

    def disconnect_cameras(self):
        """Destroy all camera reading associated threads working down to the cameras
        themselves so that the session cameras can be later reconstructed (potentially
        with additional or fewer cameras)"""

        try:
            logger.info("Attempting to shutdown monocalibrators")
            for port, monocal in self.monocalibrators.items():
                monocal.stop()
                # monocal.thread.join()

            self.monocalibrators = {}
        except (AttributeError):
            logger.warning("No monocalibrators to delete")
            pass

        try:
            logger.info("Attempting to stop stereo frame emitter")
            self.stereo_frame_emitter.stop()
            # self.stereo_frame_emitter.thread.join()

        except (AttributeError):
            logger.info("No stereo frame emitter to stop")

        try:
            logger.info("Attempting to stop stereocalibrator")
            self.stereocalibrator.stop()

        except (AttributeError):
            logger.warning("No stereocalibrator to delete.")
            pass  # don't worry if it doesn't exist

        try:
            logger.info("Attempting to stop synchronizer...")

            self.synchronizer.stop()
            del (
                self.synchronizer
            )  # important for session to know to recreate stereotools
        except (AttributeError):
            logger.warning("No synchronizer to delete")
            pass

        try:
            logger.info("Attempting to stop streams...")
            for port, stream in self.streams.items():
                stream.stop()
            self.streams = {}

            for port, cam in self.cameras.items():
                cam.capture.release()
                logger.info(f"Capture released at port {port}")
                # del cam
            # del self.cameras
            self.cameras = {}
        except (AttributeError):

            logger.warning("Unable to delete all streams...")
            pass

    def load_monocalibrators(self):
        self.corner_tracker = CornerTracker(self.charuco)

        for port, cam in self.cameras.items():
            if port in self.monocalibrators.keys():
                logger.info(
                    f"Skipping over monocalibrator creation for port {port} because it already exists."
                )
                pass  # only add if not added yet
            else:
                logger.info(f"Loading Monocalibrator for port {port}")
                self.monocalibrators[port] = MonoCalibrator(self.streams[port])

    def remove_monocalibrators(self):
        for port, monocal in self.monocalibrators.copy().items():
            logger.info(f"Attempting to stop Monocalibrator for port {port}")
            monocal.stop()
            del self.monocalibrators[port]
            logger.info(f"Successfuly stopped monocalibrator at port {port}")

    def set_active_monocalibrator(self,active_port):
        logger.info(f"Activate tracking on port {active_port} and deactivate others")
        for port,monocal in self.monocalibrators.items():
            if port == active_port:
                monocal.stream.push_to_out_q.set()
            else:
                monocal.stream.push_to_out_q.clear()

    def load_video_recorder(self):
        if hasattr(self, "synchronizer"):
            self.video_recorder = VideoRecorder(self.synchronizer)
        else:
            logger.warning("No synchronizer available to record video")

    def adjust_resolutions(self):
        """Changes the camera resolution to the value in the configuration, as
        log as it is not configured for the default resolution"""

        def adjust_res_worker(port):
            stream = self.streams[port]
            resolution = self.config[f"cam_{port}"]["resolution"]
            default_res = self.cameras[port].default_resolution

            if resolution[0] != default_res[0] or resolution[1] != default_res[1]:
                logger.info(
                    f"Beginning to change resolution at port {port} from {default_res[0:2]} to {resolution[0:2]}"
                )
                stream.change_resolution(resolution)
                logger.info(
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
            "verified_resolutions": cam.verified_resolutions,
        }

        logger.info(f"Saving camera parameters...{params}")

        self.config["cam_" + str(port)] = params
        self.update_config()


    def get_stage(self):
        if self.connected_camera_count() == 0:
            return Stage.NO_CAMERAS

        if self.calibrated_camera_count() < self.connected_camera_count():
            return Stage.UNCALIBRATED_CAMERAS

        if len(self.calibrated_camera_pairs()) == len(self.camera_pairs()):
            return Stage.STEREOCALIBRATION_DONE

        if (
            self.connected_camera_count() > 0
            and self.calibrated_camera_count() == self.connected_camera_count()
        ):
            return Stage.MONOCALIBRATED_CAMERAS


class Stage(Enum):
    NO_CAMERAS = auto()
    UNCALIBRATED_CAMERAS = auto()
    MONOCALIBRATED_CAMERAS = auto()
    STEREOCALIBRATION_IN_PROCESS = auto()
    STEREOCALIBRATION_DONE = auto()
    ORIGIN_SET = auto()


#%%
if __name__ == "__main__":
    repo = Path(str(Path(__file__)).split("calicam")[0], "calicam")
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
