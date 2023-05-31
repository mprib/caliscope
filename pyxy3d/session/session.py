# Environment for managing all created objects and the primary interface for the GUI.
import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

from PyQt6.QtCore import QObject,pyqtSignal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from enum import Enum, auto

from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.calibration.monocalibrator import MonoCalibrator
from pyxy3d.cameras.camera import Camera
from pyxy3d.cameras.synchronizer import Synchronizer
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.interface import Tracker

from pyxy3d.calibration.stereocalibrator import StereoCalibrator
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.quality_controller import QualityController

from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from pyxy3d.configurator import Configurator
from pyxy3d.cameras.live_stream import LiveStream
from pyxy3d.recording.video_recorder import VideoRecorder

# %%
MAX_CAMERA_PORT_CHECK = 10
FILTERED_FRACTION = 0.05  # by default, 5% of image points with highest reprojection error are filtered out during calibration

class SessionMode(Enum):
    """
    Note: Not currently being used for anything...if this comment remains for a few days,
    just delete this class, Mac.
    """
    Charuco = auto()
    IntrinsicCalibration = auto()
    ExtrinsicCalibration = auto()
    Recording = auto()
    PostProcessing = auto()

 
class Session(QObject):
    
    stream_tools_loaded_signal = pyqtSignal()
    
    def __init__(self, config: Configurator):
        super().__init__()
        self.config = config
        # self.folder = PurePath(directory).name
        self.path = self.config.session_path

        # this will not have anything to start, but the path
        # will be set
        self.extrinsic_calibration_xy = Path(
            self.path, "calibration", "extrinsic", "xy.csv"
        )

        # dictionaries of streaming related objects. key = port
        self.cameras = {}
        self.streams = {}
        self.stream_tools_in_process = False
        self.stream_tools_loaded = False
        # dictionaries of calibration related objects.
        self.monocalibrators = {}  # key = port
        self.active_monocalibrator = None

        # load fps for various modes
        self.fps_recording = self.config.get_fps_recording()
        self.fps_extrinsic_calibration = self.config.get_fps_extrinsic_calibration()
        self.fps_intrinsic_calibration = self.config.get_fps_intrinsic_calibration()
        
                
        self.is_recording = False

        self.charuco = self.config.get_charuco()
        self.mode = SessionMode.Charuco # default mode of session
         
    
    def set_mode(self, mode:SessionMode): 
        """
        Via this method, the frame reading behavior will be changed by the GUI. If some properties are 
        not available (i.e. synchronizer) they will be created
        """
        self.mode = mode
        self.update_streams_fps()

        match self.mode:
            case SessionMode.Charuco:
                if self.stream_tools_loaded:
                    self.synchronizer.unsubscribe_from_streams()
                    self.pause_all_monocalibrators()

            case SessionMode.PostProcessing:
                                
                if self.stream_tools_loaded:
                    self.synchronizer.unsubscribe_from_streams()
                    self.pause_all_monocalibrators()

            case SessionMode.IntrinsicCalibration:
                self.charuco_tracker = CharucoTracker(self.charuco)
                
                if not self.stream_tools_loaded:
                    self.charuco_tracker = CharucoTracker(self.charuco)
                    self.load_stream_tools()
                self.synchronizer.unsubscribe_from_streams()
                self.pause_all_monocalibrators()
                self.set_streams_charuco()
                self.set_streams_tracking(True)
                self.activate_monocalibrator(self.active_monocalibrator)

            case SessionMode.ExtrinsicCalibration:
                if not self.stream_tools_loaded:
                    self.charuco_tracker = CharucoTracker(self.charuco)
                    self.load_stream_tools()

                self.pause_all_monocalibrators()
                self.set_streams_charuco()
                self.set_streams_tracking(True)
                self.synchronizer.subscribe_to_streams()
            case SessionMode.Recording:
                if not self.stream_tools_loaded:
                    self.load_stream_tools()

                self.pause_all_monocalibrators()
                self.set_streams_tracking(False)
                self.set_streams_fps(self.fps_recording)
                self.synchronizer.subscribe_to_streams()


    def set_active_mode_fps(self, fps_target:int):
        """
        Updates the FPS used by the currently active session mode
        """
        match self.mode:
            case SessionMode.Charuco:
                pass
            case SessionMode.PostProcessing:
                pass
            case SessionMode.IntrinsicCalibration:
                self.fps_intrinsic_calibration = fps_target    
            case SessionMode.ExtrinsicCalibration:
                self.fps_extrinsic_calibration = fps_target
            case SessionMode.Recording:
                self.fps_recording = fps_target 
       
        self.update_streams_fps() 
                
    def get_active_mode_fps(self)-> int:
        fps = None
        match self.mode:
            case SessionMode.Charuco:
                pass
            case SessionMode.PostProcessing:
                pass
            case SessionMode.IntrinsicCalibration:
                fps = self.fps_intrinsic_calibration
            case SessionMode.ExtrinsicCalibration:
                fps = self.fps_extrinsic_calibration
            case SessionMode.Recording:
                fps = self.fps_recording
        return fps
    
    def update_streams_fps(self):
        active_mode_fps = self.get_active_mode_fps()
        if active_mode_fps is not None:
            for port, stream in self.streams.items():
                stream.set_fps_target(active_mode_fps)

    def set_streams_tracking(self, tracking_on:bool):
        for port, stream in self.streams.items():
            stream.set_tracking_on(tracking_on)
            
    def set_streams_charuco(self):
        
        logger.info("updating charuco in case necessary")
        self.charuco_tracker = CharucoTracker(self.charuco)
        for port, stream in self.streams.items():
            stream.update_tracker(self.charuco_tracker)

    def pause_synchronizer(self):
        logger.info("pausing synchronizer")
        self.synchronizer.unsubscribe_from_streams()

    def unpause_synchronizer(self):
        self.synchronizer.subscribe_to_streams()
        self.synchronizer.set_stream_fps(self.synchronizer.fps_target)
        
    def get_configured_camera_count(self):
        count = 0
        for key, params in self.config.dict.copy().items():
            if key.startswith("cam"):
                count += 1
        return count


    def _find_cameras(self):
        """
        Called by load_streams in the event that no cameras are returned by the configurator...
        Will populate self.cameras using multiple threads
        """

        
        def add_cam(port):
            try:
                logger.info(f"Trying port {port}")
                cam = Camera(port)
                logger.info(f"Success at port {port}")
                self.cameras[port] = cam
                self.config.save_camera(cam)
                logger.info(
                    f"Loading stream at port {port} with charuco tracker for calibration"
                )
                self.streams[port] = LiveStream(
                    cam, tracker=CharucoTracker(self.charuco)
                )
            except:
                logger.info(f"No camera at port {port}")

        with ThreadPoolExecutor() as executor:
            for i in range(0, MAX_CAMERA_PORT_CHECK):
                if i in self.cameras.keys():
                    # don't try to connect to an already connected camera
                    pass
                else:
                    executor.submit(add_cam, i)

        # remove potential stereocalibration data to start fresh
        for key in self.config.dict.copy().keys():
            if key.startswith("stereo"):
                del self.config.dict[key]

    def load_stream_tools(self):
        """
        Connects to stored cameras and creates streams with provided tracking
        Because these streams are available, the synchronizer can then be initialized
        """
        self.stream_tools_in_process = True        
        # don't bother loading cameras until you load the streams
        self.cameras = self.config.get_cameras()

        if len(self.cameras) == 0:
            self._find_cameras()
            
        for port, cam in self.cameras.items():
            if port in self.streams.keys():
                pass  # only add if not added yet
            else:
                logger.info(f"Loading Stream for port {port}")
                self.streams[port] = LiveStream(cam, tracker=self.charuco_tracker)
        
        self._adjust_resolutions()
        self._load_monocalibrators()

        # set something as a default value just to get started
        self.active_monocalibrator = min(self.monocalibrators.keys())

        self.synchronizer = Synchronizer(self.streams) # defaults to stream default fps of 6
        # recording widget becomes available when synchronizer is created
        self.stream_tools_loaded_signal.emit()
        self.stream_tools_loaded = True
        self.stream_tools_in_process = False
        
    def _load_monocalibrators(self):
        for port, cam in self.cameras.items():
            if port in self.monocalibrators.keys():
                logger.info(
                    f"Skipping over monocalibrator creation for port {port} because it already exists."
                )
                pass  # only add if not added yet
            else:
                logger.info(f"Loading Monocalibrator for port {port}")
                self.monocalibrators[port] = MonoCalibrator(self.streams[port])

    def activate_monocalibrator(self, active_port:int):
        """
        Used to make sure that only the active camera tab is reading frames during the intrinsic calibration process
        """

        logger.info(f"Activate tracking on port {active_port} and deactivate others")
        self.active_monocalibrator = active_port
        self.monocalibrators[self.active_monocalibrator].subscribe_to_stream()

    def pause_all_monocalibrators(self):
        """
        used when not actively on the camera calibration tab
        or when silencing all in preparation for activating only one
        """
        logger.info(f"Pausing all monocalibrator looping...")
        for port, monocal in self.monocalibrators.items():
            monocal.unsubscribe_to_stream()

    def start_recording(self, destination_directory: Path):
        logger.info("Initiating recording...")
        destination_directory.mkdir(parents=True, exist_ok=True)

        self.video_recorder = VideoRecorder(self.synchronizer)
        self.video_recorder.start_recording(destination_directory)
        self.is_recording = True

    def stop_recording(self):
        logger.info("Stopping recording...")
        self.video_recorder.stop_recording()
        while self.video_recorder.recording:
            logger.info("Waiting for video recorder to save out data...")
            sleep(0.5)

        self.is_recording = False

    def _adjust_resolutions(self):
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
        stereocalibrator = StereoCalibrator(
            self.config.toml_path, self.extrinsic_calibration_xy
        )
        stereocalibrator.stereo_calibrate_all(boards_sampled=10)

        self.camera_array: CameraArray = CameraArrayInitializer(
            self.config.toml_path
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
