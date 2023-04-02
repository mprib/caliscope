# Environment for managing all created objects and the primary interface for the GUI.
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os.path import exists
from pathlib import Path, PurePath
from enum import Enum, auto
from dataclasses import asdict
import numpy as np
import toml
from itertools import combinations
from time import sleep

from pyxy3d.calibration.charuco import Charuco
from pyxy3d.calibration.corner_tracker import CornerTracker
from pyxy3d.calibration.monocalibrator import MonoCalibrator
from pyxy3d.cameras.camera import Camera
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.cameras.camera_array_builder_deprecate import CameraArrayBuilder
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer


from pyxy3d.calibration.stereocalibrator import StereoCalibrator
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.quality_controller import QualityController

from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)

from pyxy3d.cameras.live_stream import LiveStream
from pyxy3d.recording.video_recorder import VideoRecorder

#%%
MAX_CAMERA_PORT_CHECK = 10
FILTERED_FRACTION = 0.05  # by default, 5% of image points with highest reprojection error are filtered out during calibration


class Session:
    def __init__(self, directory):

        self.folder = PurePath(directory).name
        self.path = directory
        self.config_path = str(Path(directory, "config.toml"))

        # this will not have anything to start, but the path
        # will be set
        self.point_data_path = Path(self.path, "point_data.csv")

        # dictionaries of streaming related objects. key = port
        self.cameras = {}
        self.streams = {}

        # dictionaries of calibration related objects.
        self.monocalibrators = {}  # key = port

        self.synchronizer_created = False

        self.load_config()
        self.load_charuco()

    def get_synchronizer(self):
        if hasattr(self, "synchronizer"):
            logger.info("returning previously created synchronizer")
            return self.synchronizer
        else:
            logger.info("creating synchronizer...")
            self.synchronizer = Synchronizer(self.streams, fps_target=6)
            return self.synchronizer

    def pause_synchronizer(self):
        logger.info("pausing synchronizer")
        self.synchronizer.unsubscribe_to_streams()

    def unpause_synchronizer(self):
        self.synchronizer.subscribe_to_streams()

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

        # alphabetize by key to maintain standardized layout
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
                count += 1
        return count

    def delete_all_cam_data(self):
        # note: needs to be a copy to avoid errors while dict changes with deletion
        for key, params in self.config.copy().items():
            if key.startswith("cam"):
                del self.config[key]
            if key.startswith("stereo"):
                del self.config[key]

        self.update_config()

    def load_cameras(self):

        # worker function that will be spun up to connect to a previously configured camera
        def add_preconfigured_cam(params):
            # try:
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

                camera = self.cameras[port]  # just for ease of reference
                camera.rotation_count = params["rotation_count"]
                camera.exposure = params["exposure"]

                # if calibration done, then populate those as well
                if "error" in params.keys():
                    logger.info(f"Camera RMSE error for port {port}: {params['error']}")
                    camera.error = params["error"]
                    camera.matrix = np.array(params["matrix"]).astype(float)
                    camera.distortions = np.array(params["distortions"]).astype(float)
                    camera.grid_count = params["grid_count"]
            # except:
            #     logger.info("Unable to connect... camera may be in use.")

        with ThreadPoolExecutor() as executor:
            for key, params in self.config.items():
                if key.startswith("cam"):
                    if params["port"] in self.streams.keys():
                        logger.info(f"Don't reload a camera at port {params['port']}")
                    else:
                        logger.info(f"Beginning to load {key} with params {params}")
                        executor.submit(add_preconfigured_cam, params)

    def set_fps_target(self, fps_target):
        if hasattr(self, "synchronizer"):
            self.synchronizer.set_fps_target(fps_target)
        else:
            logger.info(f"Attempting to change target fps in streams to {fps_target}")
            for port, stream in self.streams.items():
                stream.set_fps_target(fps_target)

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
                self.streams[port] = LiveStream(cam, charuco=self.charuco)
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
        # self.corner_tracker = CornerTracker(self.charuco)

        for port, cam in self.cameras.items():
            if port in self.monocalibrators.keys():
                logger.info(
                    f"Skipping over monocalibrator creation for port {port} because it already exists."
                )
                pass  # only add if not added yet
            else:
                logger.info(f"Loading Monocalibrator for port {port}")
                self.monocalibrators[port] = MonoCalibrator(self.streams[port])

    # This may no longer be relevant now that things are working through a subscriber model
    # def remove_monocalibrators(self):
    #     for port, monocal in self.monocalibrators.copy().items():
    #         logger.info(f"Attempting to stop Monocalibrator for port {port}")
    #         monocal.stop()
    #         del self.monocalibrators[port]
    #         logger.info(f"Successfuly stopped monocalibrator at port {port}")

    def set_active_monocalibrator(self, active_port):
        logger.info(f"Activate tracking on port {active_port} and deactivate others")
        for port, monocal in self.monocalibrators.items():
            if port == active_port:
                monocal.subscribe_to_stream()
            else:
                monocal.unsubscribe_to_stream()

    def pause_all_monocalibrators(self):
        logger.info(f"Pausing all monocalibrator looping...")
        for port, monocal in self.monocalibrators.items():
            monocal.unsubscribe_to_stream()

    def start_recording(self, destination_folder: Path = None):
        logger.info("Initiating recording...")
        if destination_folder is None:
            logger.info(f"Default to saving files in {self.path}")
            destination_folder = Path(self.path)

            self.video_recorder = VideoRecorder(self.get_synchronizer())
            self.video_recorder.start_recording(destination_folder)

    def stop_recording(self):
        logger.info("Stopping recording...")
        self.video_recorder.stop_recording()
        while self.video_recorder.recording:
            logger.info("Waiting for video recorder to save out data...")
            sleep(0.5)

    def adjust_resolutions(self):
        """Changes the camera resolution to the value in the configuration, as
        log as it is not configured for the default resolution"""

        def adjust_res_worker(port):
            stream = self.streams[port]
            size = self.config[f"cam_{port}"]["size"]
            default_size = self.cameras[port].default_resolution

            if size[0] != default_size[0] or size[1] != default_size[1]:
                logger.info(
                    f"Beginning to change resolution at port {port} from {default_size[0:2]} to {size[0:2]}"
                )
                stream.change_resolution(size)
                logger.info(
                    f"Completed change of resolution at port {port} from {default_size[0:2]} to {size[0:2]}"
                )

        with ThreadPoolExecutor() as executor:
            for port in self.cameras.keys():
                executor.submit(adjust_res_worker, port)

    def save_camera(self, port):
        def none_or_list(value):

            if value is None:
                return None
            else:
                return value.tolist()

        camera = self.cameras[port]
        params = {
            "port": camera.port,
            "size": camera.size,
            "rotation_count": camera.rotation_count,
            "error": camera.error,
            "matrix": none_or_list(camera.matrix),
            "distortions": none_or_list(camera.distortions),
            "translation": none_or_list(camera.translation),
            "rotation": none_or_list(camera.rotation),
            "exposure": camera.exposure,
            "grid_count": camera.grid_count,
            "ignore": camera.ignore,
            "verified_resolutions": camera.verified_resolutions,
        }

        self.config["cam_" + str(port)] = params
        self.update_config()

    def save_camera_array(self):
        logger.info("Saving camera array....")
        for port, camera_data in self.camera_array.cameras.items():
            camera_data = self.camera_array.cameras[port]
            params = {
                "port": camera_data.port,
                "size": camera_data.size,
                "rotation_count": camera_data.rotation_count,
                "error": camera_data.error,
                "matrix": camera_data.matrix.tolist(),
                "distortions": camera_data.distortions.tolist(),
                "exposure": camera_data.exposure,
                "grid_count": camera_data.grid_count,
                "ignore": camera_data.ignore,
                "verified_resolutions": camera_data.verified_resolutions,
                "translation": camera_data.translation.tolist(),
                "rotation": camera_data.rotation.tolist(),
            }

            self.config["cam_" + str(port)] = params

        self.update_config()

    def load_camera_array(self):
        """
        Load camera array directly from config file. The results of capture volume
        optimization and origin transformation will be reflected in this array
        which can then be the basis for future 3d point estimation
        """
        all_camera_data = {}
        for key, params in self.config.items():
            if key.startswith("cam"):
                if params["translation"] is not None:
                    port = params["port"]
                    size = params["size"]

                    logger.info(f"Adding camera {port} to calibrated camera array...")
                    cam_data = CameraData(
                        port=port,
                        size=params["size"],
                        rotation_count=params["rotation_count"],
                        error=params["error"],
                        matrix=np.array(params["matrix"]),
                        distortions=np.array(params["distortions"]),
                        exposure=params["exposure"],
                        grid_count=params["grid_count"],
                        ignore=params["ignore"],
                        verified_resolutions=params["verified_resolutions"],
                        translation=np.array(params["translation"]),
                        rotation=np.array(params["rotation"]),
                    )

                    all_camera_data[port] = cam_data

        self.camera_array = CameraArray(all_camera_data)

    def save_point_estimates(self):
        logger.info("Saving point estimates to config...")

        temp_data = asdict(self.point_estimates)
        for key, params in temp_data.items():
            temp_data[key] = params.tolist()

        self.config["point_estimates"] = temp_data

        self.update_config()

    def load_estimated_capture_volume(self):
        """
        Following capture volume optimization via bundle adjustment, or alteration
        via a transform of the origin, the entire capture volume can be reloaded
        from the config data without needing to go through the steps

        """
        self.load_point_estimates()
        self.load_camera_array()
        self.capture_volume = CaptureVolume(self.camera_array, self.point_estimates)
        # self.capture_volume.rmse = self.config["capture_volume"]["RMSE"]
        self.capture_volume.stage = self.config["capture_volume"]["stage"]

    def save_capture_volume(self):
        # self.point_estimates = self.capture_volume.point_estimates
        # self.camera_array = self.capture_volume.camera_array
        self.save_camera_array()
        self.save_point_estimates()
        self.config["capture_volume"] = {}
        # self.config["capture_volume"]["RMSE_summary"] = self.capture_volume.rmse
        self.config["capture_volume"]["stage"] = self.capture_volume.stage
        self.update_config()



    def load_point_estimates(self):
        point_estimates_dict = self.config["point_estimates"]

        for key, value in point_estimates_dict.items():
            point_estimates_dict[key] = np.array(value)

        self.point_estimates = PointEstimates(**point_estimates_dict)

    def estimate_extrinsics(self):
        """
        This is where the camera array 6 DoF is set. Many, many things are happening
        here, but they are all necessary steps of the process so I didn't want to 
        try to encapsulate any further
        """
        stereocalibrator = StereoCalibrator(self.config_path, self.point_data_path)
        stereocalibrator.stereo_calibrate_all(boards_sampled=10)

        self.camera_array: CameraArray = CameraArrayInitializer(
            self.config_path
        ).get_best_camera_array()

        self.point_estimates: PointEstimates = get_point_estimates(
            self.camera_array, self.point_data_path
        )

        self.capture_volume = CaptureVolume(self.camera_array, self.point_estimates)
        self.capture_volume.optimize()

        self.quality_controller = QualityController(self.capture_volume, self.charuco)

        logger.info(f"Removing the worst fitting {FILTERED_FRACTION*100} percent of points from the model")
        self.quality_controller.filter_point_estimates(FILTERED_FRACTION)
        self.capture_volume.optimize()
        
        self.save_capture_volume()

    ########################## STAGE ASSOCIATED METHODS #################################
    def get_stage(self):
        stage = None
        if self.connected_camera_count() == 0:
            stage = Stage.NO_CAMERAS

        elif self.calibrated_camera_count() < self.connected_camera_count():
            stage = Stage.UNCALIBRATED_CAMERAS

        elif (
            self.connected_camera_count() > 0
            and self.calibrated_camera_count() == self.connected_camera_count()
        ):
            stage = Stage.MONOCALIBRATED_CAMERAS

        elif len(self.calibrated_camera_pairs()) == len(self.camera_pairs()):
            stage = Stage.OMNICALIBRATION_DONE

        logger.info(f"Current stage of session is {stage}")
        return stage

    def connected_camera_count(self):
        """Used to keep track of where the user is in the calibration process"""
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


class Stage(Enum):
    NO_CAMERAS = auto()
    UNCALIBRATED_CAMERAS = auto()
    MONOCALIBRATED_CAMERAS = auto()
    OMNICALIBRATION_IN_PROCESS = auto()
    OMNICALIBRATION_DONE = auto()
    ORIGIN_SET = auto()


#%%
if __name__ == "__main__":
# if True:
    from pyxy3d import __root__

    config_path = Path(__root__, "tests", "demo")

    logger.info(config_path)
    logger.info("Loading session config")
    session = Session(config_path)
    # logger.info(session.get_stage())
    # session.load_camera_array()
    # session.calibrate()
    # session.save_point_estimates()
    # session.load_camera_array()
    # session.load_point_estimates()
    session.estimate_extrinsics()
    # session.build_capture_volume_from_stereopairs()
    # session.load_configured_capture_volume()
    # session.capture_volume.optimize()
    # session.capture_volume.set_origin_to_board(240, session.charuco)
    # session.save_capture_volume()
    # while session.capture_volume.rmse["overall"] > 2:
    #     session.filter_high_error(0.05)
    logger.info(
        "\n" + session.quality_controller.distance_error_summary.to_string(index=False)
    )
    # logger.info(f"Following filter of high error points, distance error is \n {session.quality_controller.distance_error}")
    # session.update_config()
    #%%%

    # create a sample dataframe

    # group the data by "board_distance" and compute the mean and percentiles

    # logger.info("Loading Cameras...")
    # session.load_cameras()

    # logger.info("Finding Cameras...")
    # session.find_cameras()
    # logger.info(session.get_stage())
    # logger.info(f"Camera pairs: {session.camera_pairs()}")
    # logger.info(f"Calibrated Camera pairs: {session.calibrated_camera_pairs()}")
    # session.disconnect_cameras()
    # logger.info(session.get_stage())
    # logger.info(f"Camera pairs: {session.camera_pairs()}")
    # logger.info(f"Calibrated Camera pairs: {session.calibrated_camera_pairs()}")

# %%
