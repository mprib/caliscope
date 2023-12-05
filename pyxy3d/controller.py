from PySide6.QtCore import QObject, Signal, QThread
import cv2
from enum import Enum, auto
from pathlib import Path
from PySide6.QtGui import QPixmap
from time import sleep
from pyxy3d.interface import Tracker
from pyxy3d.trackers.tracker_enum import TrackerEnum
from pyxy3d.post_processing.post_processor import PostProcessor
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.intrinsic_stream_manager import IntrinsicStreamManager
from pyxy3d.configurator import Configurator
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.calibration.stereocalibrator import StereoCalibrator
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.cameras.camera_array import CameraArray, CameraData
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.calibration.capture_volume.quality_controller import QualityController
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from pyxy3d.synchronized_stream_manager import SynchronizedStreamManager, read_video_properties
from pyxy3d.workspace_guide import WorkspaceGuide
from collections import OrderedDict

import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)


FILTERED_FRACTION = 0.025  # by default, 2.5% of image points with highest reprojection error are filtered out during calibration

class CalibrationStage(Enum):
    NO_INTRINSIC_VIDEO = auto()
    INTRINSIC_VIDEO_NO_INTRINSIC_CAL = auto()
    PARTIAL_INTRINSICS = auto()
    FULL_INTRINSICS_NO_EXTRINSICS = auto()
    FULLY_CALIBRATED = auto()


class Controller(QObject):
    """
    Thin layer to integrate GUI and backend
    Tracks stage of the calibration based on a variety of factors
    """

    CameraDataUpdate = Signal(int, OrderedDict)  # port, camera_display_dictionary
    IntrinsicImageUpdate = Signal(int, QPixmap)  # port, image
    IndexUpdate = Signal(int, int)  # port, frame_index
    ExtrinsicImageUpdate = Signal(dict)
    ExtrinsicCalibrationComplete = Signal()
    extrinsic_2D_complete = Signal()
    intrinsicStreamsLoaded = Signal()
    post_processing_complete = Signal()
    
    def __init__(self, workspace_dir: Path):
        super().__init__()
        self.workspace = workspace_dir
        self.config = Configurator(self.workspace)
        self.camera_count = self.config.get_camera_count()

        # streams will be used to play back recorded video with tracked markers to select frames
        self.camera_array = CameraArray({})  # empty camera array at init
        self.charuco = self.config.get_charuco()
        self.charuco_tracker = CharucoTracker(self.charuco)
        # self.camera_count = self.config.get_camera_count()  # reference to ensure that files are in place to meet user intent

        self.workspace_guide = WorkspaceGuide(self.workspace,self.camera_count)
        self.workspace_guide.intrinsic_dir.mkdir(exist_ok=True, parents=True)
        self.workspace_guide.extrinsic_dir.mkdir(exist_ok=True, parents=True)
        self.workspace_guide.recording_dir.mkdir(exist_ok=True, parents=True)

        self.capture_volume = None

    def set_camera_count(self, count:int):
        self.camera_count = count
        self.config.save_camera_count(count)
    
    def get_camera_count(self)->int:
        count = self.config.get_camera_count()
        self.camera_count = count
        return count

    
    def all_instrinsic_mp4s_available(self)->bool:
        return self.workspace_guide.all_instrinsic_mp4s_available()
   
    def all_extrinsic_mp4s_available(self)->bool:
        return self.workspace_guide.all_extrinsic_mp4s_available()


    def all_intrinsics_estimated(self)->bool:
        """
        At this point, processing extrinsics and calibrating capture volume should be allowed
        """
        return self.camera_array.all_intrinsics_calibrated()
    
    def all_extrinsics_estimated(self)->bool:
        """
        At this point, the capture volume tab should be available
        """
        cameras_good =  self.camera_array.all_extrinsics_calibrated()
        point_estimates_good = self.config.point_estimates_toml_path.exists()
        all_data_available = self.workspace_guide.all_extrinsic_mp4s_available()
        return cameras_good and point_estimates_good and all_data_available
         
    def recordings_available(self)->bool:
        return len(self.workspace_guide.valid_recording_dirs()) > 0
         
    def get_charuco_params(self) -> dict:
        return self.config.dict["charuco"]

    def update_charuco(self, charuco: Charuco):
        self.charuco = charuco
        self.config.save_charuco(self.charuco)
        self.charuco_tracker = CharucoTracker(self.charuco)

        if hasattr(self, "intrinsic_stream_manager"):
            self.intrinsic_stream_manager.update_charuco(self.charuco_tracker)
            
    def load_extrinsic_stream_manager(self):
        logger.info(f"Loading manager for streams saved to {self.extrinsic_dir}")
        self.extrinsic_stream_manager = SynchronizedStreamManager(
            recording_dir=self.extrinsic_dir,
            all_camera_data=self.camera_array.cameras,
            tracker=self.charuco_tracker,
        )

    def process_extrinsic_streams(self, fps_target=None):
        def worker():
            self.load_extrinsic_stream_manager()
            self.extrinsic_stream_manager.process_streams(fps_target=fps_target)

            output_path = Path(self.extrinsic_dir, "CHARUCO", "xy_CHARUCO.csv")
            while not output_path.exists():
                sleep(0.5)
                logger.info(
                    f"Waiting for 2D tracked points to populate at {output_path}"
                )

        self.extrinsic_process_thread = QThread()
        self.extrinsic_process_thread.run = worker
        self.extrinsic_process_thread.finished.connect(self.extrinsic_2D_complete.emit)
        self.extrinsic_process_thread.start()

        
    def load_intrinsic_stream_manager(self):
        self.intrinsic_stream_manager = IntrinsicStreamManager(
            recording_dir=self.workspace_guide.intrinsic_dir,
            cameras=self.camera_array.cameras,
            tracker=self.charuco_tracker,
        )
        
        # signal to main GUI that the Camera tab needs to be reloaded
        logger.info("Signalling that intrinsic stream manager has loaded")
        self.intrinsicStreamsLoaded.emit()

    def load_camera_array(self):
        """
        Loads self.camera_array by first populating self.all_camera_data
        """
        # load all previously configured data if it is there
        preconfigured_cameras = self.config.get_configured_camera_data()
        self.camera_array = CameraArray(preconfigured_cameras)

        # double check that no new camera associated files have been placed in the intrinsic calibration folder
        all_ports = self.workspace_guide.get_ports_in_dir(self.workspace_guide.intrinsic_dir)

        for port in all_ports:
            if port not in self.camera_array.cameras:
                self._add_camera_from_source(port)

    def _add_camera_from_source(self, port: int):
        """
        When adding source video to calibrate a camera, the function returns the camera index
        File will be transferred to workspace/calibration/intrinsic/port_{index}.mp4
        in keeping with project layout
        """
        # copy source over to standard workspace structure
        target_mp4_path = Path(self.intrinsic_dir, f"port_{port}.mp4")
        video_properties = read_video_properties(target_mp4_path)
        size = video_properties["size"]
        new_cam_data = CameraData(
            port=port,
            size=size,
        )
        self.camera_array.cameras[port] = new_cam_data
        self.config.save_camera_array(self.camera_array)
    

    def get_intrinsic_stream_frame_count(self, port):
        return self.intrinsic_stream_manager.get_frame_count(port)

    def play_intrinsic_stream(self, port):
        logger.info(f"Begin playing stream at port {port}")
        self.intrinsic_stream_manager.play_stream(port)

    def pause_intrinsic_stream(self, port):
        logger.info(f"Pausing stream at port {port}")
        self.intrinsic_stream_manager.pause_stream(port)

    def unpause_intrinsic_stream(self, port):
        logger.info(f"Unpausing stream at port {port}")
        self.intrinsic_stream_manager.unpause_stream(port)

    def stream_jump_to(self, port, frame_index):
        logger.info(f"Jump to frame {frame_index} at port {port}")
        self.intrinsic_stream_manager.stream_jump_to(port, frame_index)

    def end_stream(self, port):
        self.intrinsic_stream_manager.end_stream(port)

    def add_calibration_grid(self, port: int, frame_index: int):
        self.intrinsic_stream_manager.add_calibration_grid(port, frame_index)

    def clear_calibration_data(self, port: int):
        self.intrinsic_stream_manager.clear_calibration_data(port)

    def scale_intrinsic_stream(self, port, new_scale):
        self.intrinsic_stream_manager.frame_emitters[port].set_scale_factor(new_scale)

    def calibrate_camera(self, port):
        def worker():
            logger.info(f"Calibrating camera at port {port}")
            self.intrinsic_stream_manager.calibrate_camera(port)
            self.push_camera_data(port)
            camera_data = self.camera_array.cameras[port]
            self.config.save_camera(camera_data)
        
        self.calibrateCameraThread = QThread()
        self.calibrateCameraThread.run = worker
        self.calibrateCameraThread.start()

    def push_camera_data(self, port):
        camera_display_data = self.camera_array.cameras[port].get_display_data()
        self.CameraDataUpdate.emit(port, camera_display_data)

    def apply_distortion(self, port, undistort: bool):
        camera = self.camera_array.cameras[port]
        self.intrinsic_stream_manager.apply_distortion(camera, undistort)

    def rotate_camera(self, port, change):

        camera_data = self.camera_array.cameras[port]
        count = camera_data.rotation_count
        count += change
        if count in [-4, 4]:
            # reset if it completes a revolution
            camera_data.rotation_count = 0
        else:
            camera_data.rotation_count = count

        # note that extrinsic streams not altered.... just reload an replay
        self.intrinsic_stream_manager.set_stream_rotation(port,camera_data.rotation_count)
        
        self.push_camera_data(port)
        self.config.save_camera(camera_data)

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
        # Note that the corner distance accuracy calcs need validation...I'm not relying on them now...
        self.quality_controller = QualityController(
            self.capture_volume, charuco=self.charuco
        )

    def estimate_extrinsics(self):
        """
        This is where the camera array 6 DoF is set. Many, many things are happening
        here, but they are all necessary steps of the process so I didn't want to
        try to encapsulate any further
        """

        def worker():
            self.extrinsic_calibration_xy = Path(
                self.workspace, "calibration", "extrinsic", "CHARUCO", "xy_CHARUCO.csv"
            )

            stereocalibrator = StereoCalibrator(
                self.config.config_toml_path, self.extrinsic_calibration_xy
            )
            stereocalibrator.stereo_calibrate_all(boards_sampled=10)

            # refreshing camera array from config file
            self.camera_array: CameraArray = CameraArrayInitializer(
                self.config.config_toml_path
            ).get_best_camera_array()

            self.point_estimates: PointEstimates = get_point_estimates(
                self.camera_array, self.extrinsic_calibration_xy
            )

            self.capture_volume = CaptureVolume(self.camera_array, self.point_estimates)
            self.capture_volume.optimize()

            self.quality_controller = QualityController(
                self.capture_volume, self.charuco
            )

            logger.info(
                f"Removing the worst fitting {FILTERED_FRACTION*100} percent of points from the model"
            )
            self.quality_controller.filter_point_estimates(FILTERED_FRACTION)
            self.capture_volume.optimize()

            # saves both point estimates and camera array
            self.config.save_capture_volume(self.capture_volume)

        self.extrinsicCalibrationThread = QThread()
        self.extrinsicCalibrationThread.run = worker
        self.extrinsicCalibrationThread.finished.connect(
            self.ExtrinsicCalibrationComplete.emit
        )
        self.extrinsicCalibrationThread.start()

    def process_recordings(self, recording_path:Path, tracker_enum:TrackerEnum):
        """
        Initiates worker thread to begin post processing.
        TrackerEnum passed in so that access is given to both the tracker and the name because the name is needed for file/folder naming
        """
        def worker():
            logger.info(f"Beginning to process video files at {recording_path}")
            logger.info(f"Creating post processor for {recording_path}")
            self.post_processor = PostProcessor(self.camera_array, recording_path, tracker_enum)
            self.post_processor.create_xy()
            self.post_processor.create_xyz()

        self.process_recordings_thread = QThread()
        self.process_recordings_thread.run = worker
        self.process_recordings_thread.finished.connect(self.post_processing_complete)
        self.process_recordings_thread.start()