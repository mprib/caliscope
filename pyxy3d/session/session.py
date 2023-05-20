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
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.calibration.monocalibrator import MonoCalibrator
from pyxy3d.cameras.camera import Camera
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.interface import Tracker
from pyxy3d.trackers.tracker_enum import TrackerEnum

from pyxy3d.calibration.stereocalibrator import StereoCalibrator
from pyxy3d.calibration.capture_volume.point_estimates import (
    PointEstimates,
    load_point_estimates,
)
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.quality_controller import QualityController

from pyxy3d.cameras.camera_array import CameraArray, CameraData, get_camera_array
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from pyxy3d.configurator import Configurator
from pyxy3d.cameras.live_stream import LiveStream
from pyxy3d.recording.video_recorder import VideoRecorder

# %%
MAX_CAMERA_PORT_CHECK = 10
FILTERED_FRACTION = 0.05  # by default, 5% of image points with highest reprojection error are filtered out during calibration


class Session:
    def __init__(self, config:Configurator):
        self.config = config
        # self.folder = PurePath(directory).name
        self.path = self.config.session_path
        self.config_path = self.config.toml_path # I will know that I'm done with this branch when I can delete this...

        # this will not have anything to start, but the path
        # will be set
        self.extrinsic_calibration_xy = Path(self.path,"calibration","extrinsic", "xy.csv")

        # dictionaries of streaming related objects. key = port
        self.cameras = {}
        self.streams = {}

        # dictionaries of calibration related objects.
        self.monocalibrators = {}  # key = port
        self.synchronizer_created = False
        self.is_recording = False

        self.charuco = self.config.get_charuco()
        self.charuco_tracker = CharucoTracker(self.charuco)


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


    def get_configured_camera_count(self):
        count = 0
        for key, params in self.config.dict.copy().items():
            if key.startswith("cam"):
                count += 1
        return count


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
                self.config.save_camera(cam)
                logger.info(f"Loading stream at port {port} with charuco tracker for calibration")
                self.streams[port] = LiveStream(cam, tracker=CharucoTracker(self.charuco))
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
        # going to comment this out. This addressed a real issue that came up
        # but I'm not happy with how this does it. Going to wait to run into it 
        # again to find the best solution
        
        for key in self.config.dict.copy().keys():
            if key.startswith("stereo"):
                del self.config.dict[key]

        # self.update_config()

    def load_streams(self, tracker:Tracker = None):
        """
        Connects to stored cameras and creates streams with provided tracking
        """

        # don't bother loading cameras until you load the streams
        self.cameras = self.config.get_cameras()

        for port, cam in self.cameras.items():
            if port in self.streams.keys():
                pass  # only add if not added yet
            else:
                logger.info(f"Loading Stream for port {port}")
                self.streams[port] = LiveStream(cam, tracker=tracker)


    def load_monocalibrators(self):
        for port, cam in self.cameras.items():
            if port in self.monocalibrators.keys():
                logger.info(
                    f"Skipping over monocalibrator creation for port {port} because it already exists."
                )
                pass  # only add if not added yet
            else:
                logger.info(f"Loading Monocalibrator for port {port}")
                self.monocalibrators[port] = MonoCalibrator(self.streams[port])

    def set_active_monocalibrator(self, active_port):
        """
        Used to make sure that only the active camera tab is reading frames during the intrinsic calibration process
        """
        logger.info(f"Activate tracking on port {active_port} and deactivate others")
        for port, monocal in self.monocalibrators.items():
            if port == active_port:
                monocal.subscribe_to_stream()
            else:
                monocal.unsubscribe_to_stream()

    def pause_all_monocalibrators(self):
        """
        used when not actively on the camera calibration tab
        """
        logger.info(f"Pausing all monocalibrator looping...")
        for port, monocal in self.monocalibrators.items():
            monocal.unsubscribe_to_stream()

    def start_recording(self, destination_directory:Path):
        logger.info("Initiating recording...")
        # if destination_folder is None:
            # destination_folder = Path(self.path)
            # logger.info(f"Default to saving files in {self.path}")
        destination_directory.mkdir(parents=True, exist_ok=True)
        
        self.video_recorder = VideoRecorder(self.get_synchronizer())
        self.video_recorder.start_recording(destination_directory)
        self.is_recording = True

    def stop_recording(self):
        logger.info("Stopping recording...")
        self.video_recorder.stop_recording()
        while self.video_recorder.recording:
            logger.info("Waiting for video recorder to save out data...")
            sleep(0.5)

        self.is_recording = False

    def adjust_resolutions(self):
        """Changes the camera resolution to the value in the configuration, as
        log as it is not configured for the default resolution"""

        def adjust_res_worker(port):
            stream = self.streams[port]
            size = self.config.dict[f"cam_{port}"]["size"]
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



    def load_estimated_capture_volume(self):
        """
        Following capture volume optimization via bundle adjustment, or alteration
        via a transform of the origin, the entire capture volume can be reloaded
        from the config data without needing to go through the steps

        """
        self.point_estimates = self.config.get_point_estimates()
        self.camera_array = self.config.get_camera_array()
        self.capture_volume = CaptureVolume(self.camera_array, self.point_estimates)
        # self.capture_volume.rmse = self.config["capture_volume"]["RMSE"]
        self.capture_volume.stage = self.config.dict["capture_volume"]["stage"]
        if "origin_sync_index" in self.config.dict["capture_volume"].keys():
            self.capture_volume.origin_sync_index = self.config.dict["capture_volume"][
                "origin_sync_index"
            ]

        # QC needed to get the corner distance accuracy within the GUI
        self.quality_controller = QualityController(
            self.capture_volume, charuco=self.charuco
        )


    def estimate_extrinsics(self):
        """
        This is where the camera array 6 DoF is set. Many, many things are happening
        here, but they are all necessary steps of the process so I didn't want to
        try to encapsulate any further
        """
        stereocalibrator = StereoCalibrator(self.config_path, self.extrinsic_calibration_xy)
        stereocalibrator.stereo_calibrate_all(boards_sampled=10)

        self.camera_array: CameraArray = CameraArrayInitializer(
            self.config_path
        ).get_best_camera_array()

        self.point_estimates: PointEstimates = get_point_estimates(
            self.camera_array, self.extrinsic_calibration_xy
        )

        self.capture_volume = CaptureVolume(self.camera_array, self.point_estimates)
        self.capture_volume.optimize()

        self.quality_controller = QualityController(self.capture_volume, self.charuco)

        logger.info(
            f"Removing the worst fitting {FILTERED_FRACTION*100} percent of points from the model"
        )
        self.quality_controller.filter_point_estimates(FILTERED_FRACTION)
        self.capture_volume.optimize()

        self.config.save_capture_volume(self.capture_volume)

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
        for key in self.config.dict.keys():
            if key.startswith("cam"):
                if "error" in self.config.dict[key].keys():
                    if self.config.dict[key]["error"] is not None:
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
        for key in self.config.dict.keys():
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

