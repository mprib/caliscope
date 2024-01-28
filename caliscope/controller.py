from PySide6.QtCore import QObject, Signal, QThread
import numpy as np
import cv2
from enum import Enum, auto
from pathlib import Path
from PySide6.QtGui import QPixmap
from time import sleep, time
from caliscope.packets import Tracker
from caliscope.trackers.tracker_enum import TrackerEnum
from caliscope.post_processing.post_processor import PostProcessor
from caliscope.calibration.charuco import Charuco
from caliscope.intrinsic_stream_manager import IntrinsicStreamManager
from caliscope.configurator import Configurator
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.calibration.stereocalibrator import StereoCalibrator
from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.cameras.camera_array_initializer import CameraArrayInitializer
from caliscope.calibration.capture_volume.quality_controller import QualityController
from caliscope.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from caliscope.synchronized_stream_manager import (
    SynchronizedStreamManager,
    read_video_properties,
)
from caliscope.workspace_guide import WorkspaceGuide
from collections import OrderedDict

import caliscope.logger

logger = caliscope.logger.get(__name__)


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

    new_camera_data = Signal(int, OrderedDict)  # port, camera_display_dictionary
    capture_volume_calibrated = Signal()
    capture_volume_shifted = Signal()
    enable_inputs = Signal(int, bool)  # port, enable
    post_processing_complete = Signal()
    show_synched_frames = Signal()
    
    def __init__(self, workspace_dir: Path):
        super().__init__()
        self.workspace = workspace_dir
        self.config = Configurator(self.workspace)
        self.camera_count = self.config.get_camera_count()

        # streams will be used to play back recorded video with tracked markers to select frames
        self.camera_array = CameraArray({})  # empty camera array at init
        logger.info("Retrieving charuco from config")
        self.charuco = self.config.get_charuco()
        self.charuco_tracker = CharucoTracker(self.charuco)
        # self.camera_count = self.config.get_camera_count()  # reference to ensure that files are in place to meet user intent

        logger.info("Building workpace guide")
        self.workspace_guide = WorkspaceGuide(self.workspace, self.camera_count)
        self.workspace_guide.intrinsic_dir.mkdir(exist_ok=True, parents=True)
        self.workspace_guide.extrinsic_dir.mkdir(exist_ok=True, parents=True)
        self.workspace_guide.recording_dir.mkdir(exist_ok=True, parents=True)

        self.capture_volume = None

        # needs to exist before main widget can connect to its finished signal
        self.load_workspace_thread = QThread()
        self.calibrate_camera_threads = {}
        self.autocalibrate_threads = {}
        
    def load_workspace(self):
        def worker():
            logger.info("Assess whether to load cameras")
            if self.workspace_guide.all_instrinsic_mp4s_available():
                self.load_camera_array()
                self.load_intrinsic_stream_manager()
                self.cameras_loaded = True
            else:
                self.cameras_loaded = False

            logger.info("Assess whether to load capture volume")
            if self.all_extrinsics_estimated():
                logger.info("All extrinsics calibrated...loading capture volume")
                self.load_estimated_capture_volume()
                self.capture_volume_loaded = True
            else:
                logger.info("Not all extrinsics calibrated...not loading capture volume")
                self.capture_volume_loaded = False

        self.load_workspace_thread.run = worker
        self.load_workspace_thread.start()

    def set_camera_count(self, count: int):
        self.camera_count = count
        self.config.save_camera_count(count)

    def get_camera_count(self) -> int:
        count = self.config.get_camera_count()
        self.camera_count = count
        return count

    def all_instrinsic_mp4s_available(self) -> bool:
        return self.workspace_guide.all_instrinsic_mp4s_available()

    def all_extrinsic_mp4s_available(self) -> bool:
        return self.workspace_guide.all_extrinsic_mp4s_available()

    def all_intrinsics_estimated(self) -> bool:
        """
        At this point, processing extrinsics and calibrating capture volume should be allowed
        """
        return self.camera_array.all_intrinsics_calibrated()

    def all_extrinsics_estimated(self) -> bool:
        """
        At this point, the capture volume tab should be available
        """
        cameras_good = self.camera_array.all_extrinsics_calibrated()
        logger.info(f"All extrinsics calculated: {cameras_good}")
        point_estimates_good = self.config.point_estimates_toml_path.exists()
        logger.info(f"Point estimates available: {point_estimates_good}")
        all_data_available = self.workspace_guide.all_extrinsic_mp4s_available()
        logger.info(f"All underlying data available: {all_data_available}")
        
        return cameras_good and point_estimates_good and all_data_available

    def recordings_available(self) -> bool:
        return len(self.workspace_guide.valid_recording_dirs()) > 0

    def get_charuco_params(self) -> dict:
        return self.config.dict["charuco"]

    def update_charuco(self, charuco: Charuco):
        self.charuco = charuco
        self.config.save_charuco(self.charuco)
        self.charuco_tracker = CharucoTracker(self.charuco)

        if hasattr(self, "intrinsic_stream_manager"):
            logger.info("Updating charuco within the intrinsic stream manager")
            self.intrinsic_stream_manager.update_charuco(self.charuco_tracker)

    def load_extrinsic_stream_manager(self):
        logger.info(
            f"Loading manager for streams saved to {self.workspace_guide.extrinsic_dir}"
        )
        self.extrinsic_stream_manager = SynchronizedStreamManager(
            recording_dir=self.workspace_guide.extrinsic_dir,
            all_camera_data=self.camera_array.cameras,
            tracker=self.charuco_tracker,
        )

    # def process_extrinsic_streams(self, fps_target=None):
    #     def worker():
    #         output_path = Path(
    #             self.workspace_guide.extrinsic_dir, "CHARUCO", "xy_CHARUCO.csv"
    #         )
    #         output_path.unlink()  # make sure this doesn't exist to begin with.

    #         self.load_extrinsic_stream_manager()
    #         self.extrinsic_stream_manager.process_streams(fps_target=fps_target)

    #         logger.info(
    #             f"Processing of extrinsic calibration begun...waiting for output to populate: {output_path}"
    #         )
    #         while not output_path.exists():
    #             sleep(0.5)
    #             logger.info(
    #                 f"Waiting for 2D tracked points to populate at {output_path}"
    #             )

    def load_intrinsic_stream_manager(self):
        self.intrinsic_stream_manager = IntrinsicStreamManager(
            recording_dir=self.workspace_guide.intrinsic_dir,
            cameras=self.camera_array.cameras,
            tracker=self.charuco_tracker,
        )
        logger.info("Intrinsic stream manager has loaded")

        # signal to main GUI that the Camera tab needs to be reloaded
        # self.intrinsicStreamsLoaded.emit()

    def load_camera_array(self):
        """
        Loads self.camera_array by first populating self.all_camera_data
        """
        # load all previously configured data if it is there
        preconfigured_cameras = self.config.get_configured_camera_data()
        self.camera_array = CameraArray(preconfigured_cameras)

        # double check that no new camera associated files have been placed in the intrinsic calibration folder
        all_ports = self.workspace_guide.get_ports_in_dir(
            self.workspace_guide.intrinsic_dir
        )

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
        target_mp4_path = Path(self.workspace_guide.intrinsic_dir, f"port_{port}.mp4")
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
            if self.intrinsic_stream_manager.calibrators[port].grid_count > 0:
                self.enable_inputs.emit(port, False) 
                self.camera_array.cameras[port].erase_calibration_data()
                logger.info(f"Calibrating camera at port {port}")
                self.intrinsic_stream_manager.calibrate_camera(port)

                camera_data = self.camera_array.cameras[port]
                self.config.save_camera(camera_data)
                self.push_camera_data(port)
                self.enable_inputs.emit(port, True) 
            else:
                logger.warn("Not enough grids available to calibrate")

        self.calibrate_camera_threads[port] = QThread()
        self.calibrate_camera_threads[port].run = worker
        self.calibrate_camera_threads[port].start()

    def push_camera_data(self, port):
        logger.info(f"Pushing camera data for port {port}")
        camera_display_data = self.camera_array.cameras[port].get_display_data()
        logger.info(f"camera display data is {camera_display_data}")
        self.new_camera_data.emit(port, camera_display_data)

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
        self.intrinsic_stream_manager.set_stream_rotation(
            port, camera_data.rotation_count
        )

        self.push_camera_data(port)
        self.config.save_camera(camera_data)

    def load_estimated_capture_volume(self):
        """
        Following capture volume optimization via bundle adjustment, or alteration
        via a transform of the origin, the entire capture volume can be reloaded
        from the config data without needing to go through the steps

        """
        logger.info("Beginning to load estimated capture volume")
        self.point_estimates = self.config.get_point_estimates()
        # self.camera_array = self.config.get_camera_array()
        self.capture_volume = CaptureVolume(self.camera_array, self.point_estimates)
        logger.info("Load of capture volume complete")

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

    def calibrate_capture_volume(self):
        """
        This is where the camera array 6 DoF is set. Many, many things are happening
        here, but they are all necessary steps of the process so I didn't want to
        try to encapsulate any further
        """

        def worker():
            output_path = Path(
                self.workspace_guide.extrinsic_dir, "CHARUCO", "xy_CHARUCO.csv"
            )
            if output_path.exists():
                output_path.unlink()  # make sure this doesn't exist to begin with.

            self.load_extrinsic_stream_manager()
            self.extrinsic_stream_manager.process_streams(fps_target=100)
            logger.info(
                f"Processing of extrinsic calibration begun...waiting for output to populate: {output_path}"
            )

            logger.info("About to signal that synched frames should be shown")
            self.show_synched_frames.emit()
            
            while not output_path.exists():
                sleep(0.5)
                # moderate the frequency with which logging statements get made
                if round(time()) % 3 == 0:
                    logger.info(
                        f"Waiting for 2D tracked points to populate at {output_path}"
                    )

            # note that this processing will wait until it is complete
            # self.process_extrinsic_streams(fps_target=100)
            logger.info("Processing if extrinsic caliberation streams complete...")

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

        self.calibrate_capture_volume_thread = QThread()
        self.calibrate_capture_volume_thread.run = worker
        self.calibrate_capture_volume_thread.finished.connect(
            self.capture_volume_calibrated.emit
        )
        self.calibrate_capture_volume_thread.start()

    def process_recordings(self, recording_path: Path, tracker_enum: TrackerEnum):
        """
        Initiates worker thread to begin post processing.
        TrackerEnum passed in so that access is given to both the tracker and the name because the name is needed for file/folder naming
        """

        def worker():
            logger.info(f"Beginning to process video files at {recording_path}")
            logger.info(f"Creating post processor for {recording_path}")
            self.post_processor = PostProcessor(
                self.camera_array, recording_path, tracker_enum
            )
            self.post_processor.create_xy()
            self.post_processor.create_xyz()

        self.process_recordings_thread = QThread()
        self.process_recordings_thread.run = worker
        self.process_recordings_thread.finished.connect(self.post_processing_complete)
        self.process_recordings_thread.start()

    def rotate_capture_volume(self, direction: str):
        transformations = {
            "x+": np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
            ),
            "x-": np.array(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float
            ),
            "y+": np.array(
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
            ),
            "y-": np.array(
                [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
            ),
            "z+": np.array(
                [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),
            "z-": np.array(
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),
        }

        self.capture_volume.shift_origin(transformations[direction])
        self.capture_volume_shifted.emit()

        # don't hold up the rest of the processing just to save the capture volume
        def worker():
            self.config.save_capture_volume(self.capture_volume)

        self.rotate_capture_volume_thread = QThread()
        self.rotate_capture_volume_thread.run = worker
        self.rotate_capture_volume_thread.start()

    def set_capture_volume_origin_to_board(self, origin_index):
        self.capture_volume.set_origin_to_board(origin_index, self.charuco)
        self.capture_volume_shifted.emit()

        def worker():
            self.config.save_capture_volume(self.capture_volume)

        self.set_origin_thread = QThread()
        self.set_origin_thread.run = worker
        self.set_origin_thread.start()

    def autocalibrate(self, port, grid_count, board_threshold):
        def worker():
            self.enable_inputs.emit(port, False) 
            self.camera_array.cameras[port].erase_calibration_data()
            self.config.save_camera(self.camera_array.cameras[port])
            self.push_camera_data(port)
            logger.info(f"Initiate autocalibration of grids for port {port}")
            self.intrinsic_stream_manager.autocalibrate(
                port, grid_count, board_threshold
            )

            while self.camera_array.cameras[port].matrix is None:
                logger.info(f"Waiting for calibration to complete at port {port}")
                sleep(2)
            
            self.config.save_camera(self.camera_array.cameras[port])
            self.push_camera_data(port)
            self.intrinsic_stream_manager.stream_jump_to(port, 0)
            self.enable_inputs.emit(port, True) 

        self.autocalibrate_threads[port] = QThread()
        self.autocalibrate_threads[port].run = worker
        self.autocalibrate_threads[port].start()
            