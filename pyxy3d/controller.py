from PySide6.QtCore import QObject, Signal, Slot

from pathlib import Path

from PySide6.QtGui import QPixmap

import pyxy3d.logger
from pyxy3d.calibration.charuco import Charuco
from pyxy3d.recording.recorded_stream import RecordedStream
from pyxy3d.configurator import Configurator
from pyxy3d.trackers.charuco_tracker import CharucoTracker
from pyxy3d.interface import Tracker
from pyxy3d.calibration.stereocalibrator import StereoCalibrator
from pyxy3d.calibration.capture_volume.capture_volume import CaptureVolume
from pyxy3d.calibration.capture_volume.point_estimates import PointEstimates
from pyxy3d.cameras.camera_array import CameraArray
from pyxy3d.cameras.camera_array_initializer import CameraArrayInitializer
from pyxy3d.calibration.capture_volume.quality_controller import QualityController
from pyxy3d.calibration.capture_volume.helper_functions.get_point_estimates import (
    get_point_estimates,
)
from pyxy3d.gui.frame_emitters.playback_frame_emitter import PlaybackFrameEmitter
from pyxy3d.calibration.intrinsic_calibrator import IntrinsicCalibrator
from pyxy3d.synchronized_stream_manager import SynchronizedStreamManager
from collections import OrderedDict

logger = pyxy3d.logger.get(__name__)


FILTERED_FRACTION = 0.025  # by default, 2.5% of image points with highest reprojection error are filtered out during calibration

class Controller(QObject):
    """
    Thin layer to integrate GUI and backend
    """

    CameraDataUpdate = Signal(int, OrderedDict)  # port, camera_display_dictionary
    IntrinsicImageUpdate = Signal(int, QPixmap)  # port, image
    IndexUpdate = Signal(int, int)  # port, frame_index
    ExtrinsicImageUpdate = Signal(dict)
    ExtrinsicCalibrationComplete = Signal()
    
    def __init__(self, workspace_dir: Path):
        super().__init__()
        self.workspace = workspace_dir
        self.config = Configurator(self.workspace)

        # streams will be used to play back recorded video with tracked markers to select frames
        self.intrinsic_streams = {}
        self.frame_emitters = {}
        self.intrinsic_calibrators = {}
        self.charuco = self.config.get_charuco()
        self.charuco_tracker = CharucoTracker(self.charuco)

        self.intrinsic_source_directory = Path(
            self.workspace, "calibration", "intrinsic"
        )
        self.intrinsic_source_directory.mkdir(
            exist_ok=True, parents=True
        )  # make sure the containing directory exists

        self.extrinsic_source_directory = Path(
            self.workspace, "calibration", "extrinsic"
        )

        self.extrinsic_source_directory.mkdir(
            exist_ok=True, parents=True
        )  # make sure the containing directory exists

        self.capture_volume = None
        
        
    def get_intrinsic_stream_frame_count(self, port):
        start_frame_index = self.intrinsic_streams[port].start_frame_index
        last_frame_index = self.intrinsic_streams[port].last_frame_index

        return last_frame_index - start_frame_index + 1

    def get_charuco_params(self) -> dict:
        return self.config.dict["charuco"]

    def update_charuco(self, charuco: Charuco):
        self.charuco = charuco
        self.charuco_tracker = CharucoTracker(self.charuco)
        self.config.save_charuco(self.charuco)

        for port, stream in self.intrinsic_streams.items():
            logger.info(f"Updating tracker for stream at port {port}")
            stream.tracker = self.charuco_tracker
            # stream.set_tracking_on(True)

    def process_extrinsic_streams(self, fps_target = None):

        
        self.sync_stream_manager = SynchronizedStreamManager(
            recording_dir=self.extrinsic_source_directory,
            all_camera_data=self.all_camera_data,
            tracker=self.charuco_tracker,
        )
        self.sync_stream_manager.process_streams(fps_target=fps_target)

    def load_intrinsic_streams(self):
        for port, camera_data in self.all_camera_data.items():
            # data storage convention defined here
            source_file = Path(self.intrinsic_source_directory, f"port_{port}.mp4")
            logger.info(f"Loading stream associated with source file at {source_file}")

            rotation_count = camera_data.rotation_count

            stream = RecordedStream(
                directory=self.intrinsic_source_directory,
                port=port,
                rotation_count=rotation_count,
                tracker=self.charuco_tracker,
                break_on_last=False,
            )

            self.frame_emitters[port] = PlaybackFrameEmitter(stream)
            self.frame_emitters[port].start()
            self.frame_emitters[port].ImageBroadcast.connect(
                self.broadcast_frame_update
            )
            self.frame_emitters[port].FrameIndexBroadcast.connect(
                self.broadcast_index_update
            )

            self.intrinsic_streams[port] = stream
            self.intrinsic_calibrators[port] = IntrinsicCalibrator(camera_data, stream)
            logger.info(f"Loading recorded stream stored in {source_file}")

    @Slot(int, QPixmap)
    def broadcast_frame_update(self, port, pixmap):
        logger.info(f"Broadcast frame update from port {port}")
        self.IntrinsicImageUpdate.emit(port, pixmap)

    @Slot(int, int)
    def broadcast_index_update(self, port, index):
        logger.info(f"Broadcast index update from port {port}")
        self.IndexUpdate.emit(port, index)

    def load_camera_array(self):
        """
        Loads self.camera_array by first populating self.all_camera_data
        """
        # load all previously configured data if it is there
        self.all_camera_data = self.config.get_configured_camera_data()
        
        # double check that no new camera associated files have been placed in the intrinsic calibration folder
        all_ports = self.config.get_all_source_camera_ports()
        for port in all_ports:
            if port not in self.all_camera_data:
                self.add_camera_from_source(port)
        self.camera_array = CameraArray(self.all_camera_data)
        
    def add_camera_from_source(self, port: int):
        """
        When adding source video to calibrate a camera, the function returns the camera index
        File will be transferred to workspace/calibration/intrinsic/port_{index}.mp4
        in keeping with project layout
        """
        # copy source over to standard workspace structure
        new_cam_data = self.config.get_camera_from_source(port)
        self.all_camera_data[port] = new_cam_data
        self.config.save_all_camera_data(self.all_camera_data)

    def set_current_tracker(self, tracker: Tracker = None):
        self.tracker = tracker

    def play_intrinsic_stream(self, port):
        logger.info(f"Begin playing stream at port {port}")
        self.intrinsic_streams[port].play_video()

    def pause_intrinsic_stream(self, port):
        logger.info(f"Pausing stream at port {port}")
        self.intrinsic_streams[port].pause()

    def unpause_intrinsic_stream(self, port):
        logger.info(f"Unpausing stream at port {port}")
        self.intrinsic_streams[port].unpause()

    def stream_jump_to(self, port, frame_index):
        logger.info(f"Jump to frame {frame_index} at port {port}")
        self.intrinsic_streams[port].jump_to(frame_index)

    def end_stream(self, port):
        self.intrinsic_streams[port].stop_event.set()
        self.unpause_intrinsic_stream(port)

    def add_calibration_grid(self, port: int, frame_index: int):
        intr_calib = self.intrinsic_calibrators[port]
        intr_calib.add_calibration_frame_indices(frame_index)
        new_ids = intr_calib.all_ids[frame_index]
        new_img_loc = intr_calib.all_img_loc[frame_index]
        self.frame_emitters[port].add_to_grid_history(new_ids, new_img_loc)

    def clear_calibration_data(self, port: int):
        intr_calib = self.intrinsic_calibrators[port]
        intr_calib.clear_calibration_data()
        self.frame_emitters[port].initialize_grid_capture_history()

    def calibrate_camera(self, port):
        logger.info(f"Calibrating camera at port {port}")
        self.intrinsic_calibrators[port].calibrate_camera()
        logger.info(f"{self.all_camera_data[port]}")
        self.push_camera_data(port)
        camera_data = self.all_camera_data[port]
        self.config.save_camera(camera_data)
        # camera_display_data = self.all_camera_data[port].get_display_data()
        # self.CameraDataUpdate.emit(port,camera_display_data)

    def push_camera_data(self, port):
        camera_display_data = self.all_camera_data[port].get_display_data()
        self.CameraDataUpdate.emit(port, camera_display_data)

    def apply_distortion(self, port, undistort: bool):
        camera_data = self.all_camera_data[port]
        emitter = self.frame_emitters[port]
        emitter.update_distortion_params(
            undistort, camera_data.matrix, camera_data.distortions
        )

    def rotate_camera(self, port, change):
        camera_data = self.all_camera_data[port]

        count = camera_data.rotation_count
        count += change
        if count in [-4, 4]:
            # reset if it completes a revolution
            camera_data.rotation_count = 0
        else:
            camera_data.rotation_count = count

        stream = self.intrinsic_streams[port]
        stream.rotation_count = camera_data.rotation_count
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
        self.quality_controller = QualityController(
            self.capture_volume, charuco=self.charuco
        )

    def estimate_extrinsics(self):
        """
        This is where the camera array 6 DoF is set. Many, many things are happening
        here, but they are all necessary steps of the process so I didn't want to
        try to encapsulate any further
        """
        self.extrinsic_calibration_xy = Path(
            self.workspace, "calibration", "extrinsic", "CHARUCO", "xy_CHARUCO.csv"
        )

        stereocalibrator = StereoCalibrator(
            self.config.config_toml_path, self.extrinsic_calibration_xy
        )
        stereocalibrator.stereo_calibrate_all(boards_sampled=10)

        self.camera_array: CameraArray = CameraArrayInitializer(
            self.config.config_toml_path
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

        # saves both point estimates and camera array
        self.config.save_capture_volume(self.capture_volume)

        self.ExtrinsicCalibrationComplete.emit()