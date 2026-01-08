import logging
from collections import OrderedDict
from pathlib import Path
from time import time
from datetime import datetime


from PySide6.QtCore import QObject, Signal

from caliscope.task_manager import TaskHandle, TaskManager

from caliscope.core.capture_volume.capture_volume import CaptureVolume
from caliscope.core.capture_volume.quality_controller import QualityController
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.core.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.repositories import (
    CameraArrayRepository,
    CaptureVolumeRepository,
    CharucoRepository,
    ProjectSettingsRepository,
)
from caliscope.managers.intrinsic_stream_manager import IntrinsicStreamManager
from caliscope.post_processing.post_processor import PostProcessor
from caliscope.managers.synchronized_stream_manager import (
    SynchronizedStreamManager,
    read_video_properties,
)
from caliscope.core.point_data import ImagePoints
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.trackers.tracker_enum import TrackerEnum
from caliscope.workspace_guide import WorkspaceGuide

logger = logging.getLogger(__name__)


FILTERED_FRACTION = (
    0.025  # by default, 2.5% of image points with highest reprojection error are filtered out during calibration
)


class Controller(QObject):
    """
    Thin integration layer between GUI and backend domain logic.

    The Controller orchestrates the calibration workflow by coordinating between
    managers (data persistence), stream managers (video processing), and domain
    objects (CameraArray, CaptureVolume). It maintains no business logic itself
    beyond workflow state management.

    All data access is delegated to typed manager classes, eliminating coupling
    between the GUI and persistence implementation details.
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

        # Initialize repositories with explicit file paths
        self.settings_repository = ProjectSettingsRepository(workspace_dir / "project_settings.toml")
        self.camera_repository = CameraArrayRepository(workspace_dir / "camera_array.toml")
        self.charuco_repository = CharucoRepository(workspace_dir / "charuco.toml")
        self.capture_volume_repository = CaptureVolumeRepository(workspace_dir)

        # Initialize project files if they don't exist
        self._initialize_project_files()

        self.camera_count = self.settings_repository.get_camera_count()

        # streams will be used to play back recorded video with tracked markers to select frames
        self.camera_array = CameraArray({})  # empty camera array at init

        logger.info("Loading charuco from manager")
        self.charuco = self.charuco_repository.load()
        self.charuco_tracker = CharucoTracker(self.charuco)

        logger.info("Building workspace guide")
        self.workspace_guide = WorkspaceGuide(self.workspace)
        self.workspace_guide.intrinsic_dir.mkdir(exist_ok=True, parents=True)
        self.workspace_guide.extrinsic_dir.mkdir(exist_ok=True, parents=True)
        self.workspace_guide.recording_dir.mkdir(exist_ok=True, parents=True)

        self.capture_volume = None
        self.cameras_loaded = False
        self.capture_volume_loaded = False

        # Centralized task management for background operations
        self.task_manager = TaskManager(parent=self)

    def _initialize_project_files(self):
        """Create default project files if they don't exist."""
        logger.info("Checking for existing project files...")

        # Project settings (always create/update to ensure required fields exist)
        if not self.settings_repository.path.exists():
            logger.info("Creating default project settings")
            self.settings_repository.save(
                {
                    "creation_date": datetime.now().isoformat(),
                    "camera_count": 0,
                    "save_tracked_points_video": True,
                    "fps_sync_stream_processing": 100,
                }
            )

        # Charuco board (create default if missing)
        if not self.charuco_repository.path.exists():
            logger.info("Creating default charuco board")
            default_charuco = Charuco(4, 5, 11, 8.5, square_size_overide_cm=5.4)
            self.charuco_repository.save(default_charuco)

        # Camera array (create empty if missing)
        if not self.camera_repository.path.exists():
            logger.info("Creating empty camera array")
            empty_array = CameraArray({})
            self.camera_repository.save(empty_array)

    def load_workspace(self) -> TaskHandle:
        """Asynchronously load workspace state on startup.

        Returns:
            TaskHandle for connecting completion callbacks.
        """

        def worker(_token, _handle):
            logger.info("Assessing whether to load cameras")
            if self.workspace_guide.all_instrinsic_mp4s_available(self.camera_count):
                self.load_camera_array()
                self.load_intrinsic_stream_manager()
                self.cameras_loaded = True
            else:
                self.cameras_loaded = False

            logger.info("Assessing whether to load capture volume")
            if self.all_extrinsics_estimated():
                logger.info("All extrinsics calibrated...loading capture volume")
                self.load_estimated_capture_volume()
                self.capture_volume_loaded = True
            else:
                logger.info("Not all extrinsics calibrated...not loading capture volume")
                self.capture_volume_loaded = False

        return self.task_manager.submit(worker, name="load_workspace")

    def set_camera_count(self, count: int):
        """Update camera count in project settings."""
        self.camera_count = count
        self.settings_repository.set_camera_count(count)

    def get_camera_count(self) -> int:
        """Get current camera count from settings."""
        count = self.settings_repository.get_camera_count()
        self.camera_count = count
        return count

    def all_instrinsic_mp4s_available(self) -> bool:
        """Check if all intrinsic calibration videos are present."""
        return self.workspace_guide.all_instrinsic_mp4s_available(self.camera_count)

    def all_extrinsic_mp4s_available(self) -> bool:
        """Check if all extrinsic calibration videos are present."""
        return self.workspace_guide.all_extrinsic_mp4s_available(self.camera_count)

    def all_intrinsics_estimated(self) -> bool:
        """
        Check if all cameras have complete intrinsic calibration.

        At this point, processing extrinsics and calibrating capture volume should be allowed.
        """
        return self.camera_array.all_intrinsics_calibrated()

    def all_extrinsics_estimated(self) -> bool:
        """
        Check if full extrinsic calibration is complete.

        At this point, the capture volume tab should be available.
        """
        cameras_good = self.camera_array.all_extrinsics_calibrated()
        logger.info(f"All extrinsics calculated: {cameras_good}")

        point_estimates_good = self.capture_volume_repository.point_estimates_path.exists()
        logger.info(f"Point estimates available: {point_estimates_good}")

        all_data_available = self.workspace_guide.all_extrinsic_mp4s_available(self.camera_count)
        logger.info(f"All underlying data available: {all_data_available}")

        return cameras_good and point_estimates_good and all_data_available

    def recordings_available(self) -> bool:
        """Check if any valid recording directories exist."""
        return len(self.workspace_guide.valid_recording_dirs()) > 0

    def update_charuco(self, charuco: Charuco):
        """
        Update charuco board definition and persist to disk.

        Also updates the charuco tracker used by stream managers.
        """
        self.charuco = charuco
        self.charuco_repository.save(self.charuco)
        self.charuco_tracker = CharucoTracker(self.charuco)

        if hasattr(self, "intrinsic_stream_manager"):
            logger.info("Updating charuco within the intrinsic stream manager")
            self.intrinsic_stream_manager.update_charuco(self.charuco_tracker)

    def get_charuco_params(self) -> dict:
        return self.charuco.__dict__

    def load_extrinsic_stream_manager(self):
        """Initialize stream manager for extrinsic calibration videos."""
        logger.info(f"Loading manager for streams saved to {self.workspace_guide.extrinsic_dir}")
        self.extrinsic_stream_manager = SynchronizedStreamManager(
            recording_dir=self.workspace_guide.extrinsic_dir,
            all_camera_data=self.camera_array.cameras,
            tracker=self.charuco_tracker,
        )

    def load_intrinsic_stream_manager(self):
        """Initialize stream manager for intrinsic calibration videos."""
        self.intrinsic_stream_manager = IntrinsicStreamManager(
            recording_dir=self.workspace_guide.intrinsic_dir,
            cameras=self.camera_array.cameras,
            tracker=self.charuco_tracker,
        )
        logger.info("Intrinsic stream manager has loaded")

    def load_camera_array(self):
        """
        Load camera array from persistence and detect new cameras from video files.

        Any cameras discovered in the intrinsic directory that aren't in the
        saved array will be added automatically.
        """
        self.camera_array = self.camera_repository.load()

        # double check that no new camera associated files have been placed in the intrinsic calibration folder
        all_ports = self.workspace_guide.get_ports_in_dir(self.workspace_guide.intrinsic_dir)

        for port in all_ports:
            if port not in self.camera_array.cameras:
                self._add_camera_from_source(port)

    def _add_camera_from_source(self, port: int):
        """
        Add a new camera discovered from video file in intrinsic directory.

        File will be transferred to workspace/calibration/intrinsic/port_{index}.mp4
        in keeping with project layout.
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
        self.camera_repository.save(self.camera_array)

    def get_intrinsic_stream_frame_count(self, port):
        """Get frame count for intrinsic stream at given port."""
        return self.intrinsic_stream_manager.get_frame_count(port)

    def play_intrinsic_stream(self, port):
        """Begin playback of intrinsic stream."""
        logger.info(f"Begin playing stream at port {port}")
        self.intrinsic_stream_manager.play_stream(port)

    def pause_intrinsic_stream(self, port):
        """Pause intrinsic stream playback."""
        logger.info(f"Pausing stream at port {port}")
        self.intrinsic_stream_manager.pause_stream(port)

    def unpause_intrinsic_stream(self, port):
        """Resume paused intrinsic stream."""
        logger.info(f"Unpausing stream at port {port}")
        self.intrinsic_stream_manager.unpause_stream(port)

    def stream_jump_to(self, port, frame_index):
        """Seek intrinsic stream to specific frame."""
        logger.info(f"Jump to frame {frame_index} at port {port}")
        self.intrinsic_stream_manager.stream_jump_to(port, frame_index)

    def end_stream(self, port):
        """Terminate stream playback and release resources."""
        self.intrinsic_stream_manager.end_stream(port)

    def add_calibration_grid(self, port: int, frame_index: int):
        """Add calibration grid point at specific frame."""
        self.intrinsic_stream_manager.add_calibration_grid(port, frame_index)

    def clear_calibration_data(self, port: int):
        """Clear all calibration data for a camera."""
        self.intrinsic_stream_manager.clear_calibration_data(port)

    def scale_intrinsic_stream(self, port, new_scale):
        """Adjust display scale of intrinsic stream."""
        self.intrinsic_stream_manager.frame_emitters[port].set_scale_factor(new_scale)

    def calibrate_camera(self, port):
        """Calibrate single camera in worker thread."""

        def worker(_token, _handle):
            if self.intrinsic_stream_manager.calibrators[port].grid_count > 0:
                self.enable_inputs.emit(port, False)
                self.camera_array.cameras[port].erase_calibration_data()
                logger.info(f"Calibrating camera at port {port}")
                self.intrinsic_stream_manager.calibrate_camera(port)

                camera_data = self.camera_array.cameras[port]
                self.camera_repository.save_camera(camera_data)
                self.push_camera_data(port)
                self.enable_inputs.emit(port, True)
            else:
                logger.warning("Not enough grids available to calibrate")

        self.task_manager.submit(worker, name=f"calibrate_camera_{port}")

    def push_camera_data(self, port):
        """Emit signal with updated camera display data."""
        logger.info(f"Pushing camera data for port {port}")
        camera_display_data = self.camera_array.cameras[port].get_display_data()
        logger.info(f"camera display data is {camera_display_data}")
        self.new_camera_data.emit(port, camera_display_data)

    def apply_distortion(self, port: int, undistort: bool):
        """Toggle distortion correction for stream display."""
        self.intrinsic_stream_manager.apply_distortion(port, undistort)

    def rotate_camera(self, port, change):
        """Adjust camera rotation count and persist."""
        camera_data = self.camera_array.cameras[port]
        count = camera_data.rotation_count
        count += change
        if count in [-4, 4]:
            # reset if it completes a revolution
            camera_data.rotation_count = 0
        else:
            camera_data.rotation_count = count

        # note that extrinsic streams not altered.... just reload an replay
        self.intrinsic_stream_manager.set_stream_rotation(port, camera_data.rotation_count)

        self.push_camera_data(port)
        self.camera_repository.save_camera(camera_data)

    def load_estimated_capture_volume(self):
        """
        Load capture volume data from persistence layer.

        This method coordinates loading of point estimates and metadata,
        then reconstructs the CaptureVolume domain object.
        """
        logger.info("Beginning to load estimated capture volume")

        # Load point estimates from dedicated manager
        self.point_estimates = self.capture_volume_repository.load_point_estimates()
        self.capture_volume = CaptureVolume(self.camera_array, self.point_estimates)

        # Load metadata and apply to capture volume
        metadata = self.capture_volume_repository.load_metadata()
        self.capture_volume.stage = metadata.get("stage")
        self.capture_volume.origin_sync_index = metadata.get("origin_sync_index")

        logger.info("Load of capture volume complete")

        # QC needed to get the corner distance accuracy within the GUI
        self.quality_controller = QualityController(self.capture_volume, charuco=self.charuco)

    def calibrate_capture_volume(self) -> TaskHandle:
        """Perform full extrinsic calibration in worker thread.

        Returns:
            TaskHandle for connecting completion callbacks.
        """

        def worker(token, _handle):
            output_path = Path(self.workspace_guide.extrinsic_dir, "CHARUCO", "xy_CHARUCO.csv")
            if output_path.exists():
                output_path.unlink()  # ensure clean start

            self.load_extrinsic_stream_manager()

            # Get processing settings from project configuration
            include_video = self.settings_repository.get_save_tracked_points_video()
            fps_target = self.settings_repository.get_fps_sync_stream_processing()

            self.extrinsic_stream_manager.process_streams(fps_target=fps_target, include_video=include_video)
            logger.info(f"Processing of extrinsic calibration begun...waiting for output to populate: {output_path}")

            logger.info("About to signal that synched frames should be shown")
            self.show_synched_frames.emit()

            # Cancellable wait for tracked points
            while not output_path.exists():
                if token.sleep_unless_cancelled(0.5):
                    return  # User cancelled
                # moderate the frequency with which logging statements get made
                if round(time()) % 3 == 0:
                    logger.info(f"Waiting for 2D tracked points to populate at {output_path}")

            if token.is_cancelled:
                return

            logger.info("Processing of extrinsic calibration streams complete...")

            self.extrinsic_calibration_xy = Path(
                self.workspace, "calibration", "extrinsic", "CHARUCO", "xy_CHARUCO.csv"
            )

            image_points = ImagePoints.from_csv(self.extrinsic_calibration_xy)

            # initialize estimated extrinsics from paired poses
            paired_pose_network = build_paired_pose_network(image_points, self.camera_array)
            paired_pose_network.apply_to(self.camera_array)

            world_points = image_points.triangulate(self.camera_array)

            self.point_estimates = world_points.to_point_estimates(image_points, self.camera_array)

            if token.is_cancelled:
                return

            # Bundle adjustment (can't interrupt mid-call)
            self.capture_volume = CaptureVolume(self.camera_array, self.point_estimates)
            self.capture_volume.optimize()

            self.quality_controller = QualityController(self.capture_volume, self.charuco)

            logger.info(f"Removing the worst fitting {FILTERED_FRACTION * 100} percent of points from the model")
            self.quality_controller.filter_point_estimates(FILTERED_FRACTION)

            if token.is_cancelled:
                return

            self.capture_volume.optimize()
            self.capture_volume_loaded = True

            # Save complete capture volume state
            self.camera_repository.save(self.camera_array)
            self.capture_volume_repository.save_capture_volume(self.capture_volume)

        handle = self.task_manager.submit(worker, name="calibrate_capture_volume")
        handle.completed.connect(lambda _: self.capture_volume_calibrated.emit())
        return handle

    def process_recordings(self, recording_path: Path, tracker_enum: TrackerEnum) -> TaskHandle:
        """
        Initiate post-processing of recorded video in worker thread.

        Args:
            recording_path: Directory containing synchronized video recordings
            tracker_enum: Tracker type to use for landmark detection

        Returns:
            TaskHandle for connecting completion callbacks.
        """

        def worker(token, _handle):
            logger.info(f"Beginning to process video files at {recording_path}")
            logger.info(f"Creating post processor for {recording_path}")
            self.post_processor = PostProcessor(self.camera_array, recording_path, tracker_enum)

            # Get processing settings from project configuration
            include_video = self.settings_repository.get_save_tracked_points_video()
            fps_target = self.settings_repository.get_fps_sync_stream_processing()

            # Pass token for cancellation support
            if not self.post_processor.create_xy(include_video=include_video, fps_target=fps_target, token=token):
                return  # Cancelled

            self.post_processor.create_xyz()

        handle = self.task_manager.submit(worker, name="process_recordings")
        handle.completed.connect(lambda _: self.post_processing_complete.emit())
        return handle

    def rotate_capture_volume(self, direction: str):
        """Rotate capture volume and persist in background thread."""
        self.capture_volume.rotate(direction)
        self.capture_volume_shifted.emit()

        def worker(_token, _handle):
            self.camera_repository.save(self.camera_array)
            self.capture_volume_repository.save_capture_volume(self.capture_volume)

        self.task_manager.submit(worker, name="rotate_capture_volume")

    def set_capture_volume_origin_to_board(self, origin_index):
        """Set world origin and persist in background thread."""
        self.capture_volume.set_origin_to_board(origin_index, self.charuco)
        self.capture_volume_shifted.emit()

        def worker(_token, _handle):
            self.camera_repository.save(self.camera_array)
            self.capture_volume_repository.save_capture_volume(self.capture_volume)

        self.task_manager.submit(worker, name="set_capture_volume_origin")

    def autocalibrate(self, port, grid_count, board_threshold):
        """Auto-calibrate camera in worker thread."""

        def worker(token, _handle):
            self.enable_inputs.emit(port, False)
            self.camera_array.cameras[port].erase_calibration_data()
            self.camera_repository.save_camera(self.camera_array.cameras[port])
            self.push_camera_data(port)

            logger.info(f"Initiate autocalibration of grids for port {port}")
            self.intrinsic_stream_manager.autocalibrate(port, grid_count, board_threshold)

            while self.camera_array.cameras[port].matrix is None:
                logger.info(f"Waiting for calibration to complete at port {port}")
                if token.sleep_unless_cancelled(2):
                    # User cancelled - re-enable inputs and exit
                    self.enable_inputs.emit(port, True)
                    return

            self.camera_repository.save_camera(self.camera_array.cameras[port])
            self.push_camera_data(port)
            self.intrinsic_stream_manager.stream_jump_to(port, 0)
            self.enable_inputs.emit(port, True)

        self.task_manager.submit(worker, name=f"autocalibrate_{port}")
