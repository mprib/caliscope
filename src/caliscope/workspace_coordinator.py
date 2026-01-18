import logging
from collections import OrderedDict
from pathlib import Path
from time import time
from datetime import datetime
from typing import Literal


from PySide6.QtCore import QObject, Signal

from caliscope.task_manager import TaskHandle, TaskManager

from caliscope.core.capture_volume.capture_volume import CaptureVolume
from caliscope.core.capture_volume.quality_controller import QualityController
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.core.charuco import Charuco
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.calibrate_intrinsics import IntrinsicCalibrationOutput, IntrinsicCalibrationReport
from caliscope.repositories import (
    CameraArrayRepository,
    CaptureVolumeRepository,
    CharucoRepository,
    ProjectSettingsRepository,
)
from caliscope.repositories.point_data_bundle_repository import PointDataBundleRepository
from caliscope.core.point_data_bundle import PointDataBundle
from caliscope.persistence import PersistenceError
from caliscope.repositories.intrinsic_report_repository import IntrinsicReportRepository
from caliscope.post_processing.post_processor import PostProcessor
from caliscope.managers.synchronized_stream_manager import (
    SynchronizedStreamManager,
    read_video_properties,
)
from caliscope.core.point_data import ImagePoints
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.trackers.tracker_enum import TrackerEnum
from caliscope.workspace_guide import WorkspaceGuide
from caliscope.gui.presenters.intrinsic_calibration_presenter import IntrinsicCalibrationPresenter
from caliscope.packets import PointPacket

logger = logging.getLogger(__name__)


FILTERED_FRACTION = (
    0.025  # by default, 2.5% of image points with highest reprojection error are filtered out during calibration
)


class WorkspaceCoordinator(QObject):
    """
    Application-level coordinator for a calibration workspace.

    Orchestrates the calibration workflow by coordinating between repositories
    (persistence), stream managers (video processing), and domain objects
    (CameraArray, CaptureVolume). Maintains no business logic itself beyond
    workflow state management.

    This is session-scoped to a workspace directory. All data access is delegated
    to typed repository classes, eliminating coupling between the GUI and
    persistence implementation details.
    """

    new_camera_data = Signal(int, OrderedDict)  # port, camera_display_dictionary
    capture_volume_calibrated = Signal()
    charuco_changed = Signal()  # Emitted when charuco board config is updated
    capture_volume_shifted = Signal()
    bundle_updated = Signal()  # PointDataBundle changed (new system, parallel to CaptureVolume)
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
        self.intrinsic_report_repository = IntrinsicReportRepository(
            workspace_dir / "calibration" / "intrinsic" / "reports"
        )

        # PointDataBundle (new system, parallel to CaptureVolume)
        # These two systems are independent during migration - do not mix them
        self.bundle_repository = PointDataBundleRepository(workspace_dir / "calibration" / "extrinsic" / "CHARUCO")
        self._point_data_bundle: PointDataBundle | None = None

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

        # In-memory cache of intrinsic calibration data (by port)
        # These enable overlay restoration when switching between cameras
        self._intrinsic_reports: dict[int, IntrinsicCalibrationReport] = {}
        # Session-only cache of collected points for overlay rendering
        # Not persisted to disk - lost on app restart
        self._intrinsic_points: dict[int, list[tuple[int, PointPacket]]] = {}

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
        Emits charuco_changed signal so dependent components can refresh.
        """
        self.charuco = charuco
        self.charuco_repository.save(self.charuco)
        self.charuco_tracker = CharucoTracker(self.charuco)
        self.charuco_changed.emit()

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

    def load_camera_array(self):
        """
        Load camera array from persistence and detect new cameras from video files.

        Any cameras discovered in the intrinsic directory that aren't in the
        saved array will be added automatically. Also loads any persisted
        intrinsic calibration reports for overlay restoration.
        """
        self.camera_array = self.camera_repository.load()

        # double check that no new camera associated files have been placed in the intrinsic calibration folder
        all_ports = self.workspace_guide.get_ports_in_dir(self.workspace_guide.intrinsic_dir)

        for port in all_ports:
            if port not in self.camera_array.cameras:
                self._add_camera_from_source(port)

        # Load any persisted intrinsic reports for overlay restoration
        self._intrinsic_reports = self.intrinsic_report_repository.load_all()
        if self._intrinsic_reports:
            logger.info(f"Loaded intrinsic reports for ports: {list(self._intrinsic_reports.keys())}")

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

    def push_camera_data(self, port):
        """Emit signal with updated camera display data."""
        logger.info(f"Pushing camera data for port {port}")
        camera_display_data = self.camera_array.cameras[port].get_display_data()
        logger.info(f"camera display data is {camera_display_data}")
        self.new_camera_data.emit(port, camera_display_data)

    def create_intrinsic_presenter(self, port: int) -> IntrinsicCalibrationPresenter:
        """Create presenter for intrinsic calibration of a single camera.

        Factory method that assembles the presenter with all required dependencies.
        The caller is responsible for connecting signals and managing presenter lifecycle.

        If a previous calibration exists (report in cache), it's passed to the presenter
        for overlay restoration. Collected points are only available during the session
        (not after app restart).

        Raises:
            ValueError: If port is not in camera_array or intrinsic video doesn't exist.
        """
        if port not in self.camera_array.cameras:
            raise ValueError(f"No camera data for port {port}")

        camera = self.camera_array.cameras[port]
        video_path = self.workspace_guide.intrinsic_dir / f"port_{port}.mp4"

        if not video_path.exists():
            raise ValueError(f"No intrinsic video for port {port}")

        # Get cached data for overlay restoration
        report = self._intrinsic_reports.get(port)
        collected_points = self._intrinsic_points.get(port)

        return IntrinsicCalibrationPresenter(
            camera=camera,
            video_path=video_path,
            tracker=self.charuco_tracker,
            task_manager=self.task_manager,
            restored_report=report,
            restored_points=collected_points,
        )

    def persist_intrinsic_calibration(
        self,
        output: IntrinsicCalibrationOutput,
        collected_points: list[tuple[int, PointPacket]] | None = None,
    ) -> None:
        """Persist intrinsic calibration result to ground truth.

        Updates the in-memory camera array and saves to disk. Also caches
        the calibration report for overlay restoration when switching cameras
        and persists it to disk for reload on app restart.
        Emits new_camera_data signal so UI components can update their display.

        Args:
            output: Complete calibration output with camera and report
            collected_points: Optional list of (frame_index, PointPacket) for
                overlay restoration during session. Not persisted to disk.
        """
        port = output.camera.port

        # Update camera in array and save
        self.camera_array.cameras[port] = output.camera
        self.camera_repository.save(self.camera_array)

        # Cache report for overlay restoration and save to disk
        self._intrinsic_reports[port] = output.report
        self.intrinsic_report_repository.save(port, output.report)

        # Cache collected points for session-only overlay restoration
        if collected_points is not None:
            self._intrinsic_points[port] = collected_points

        logger.info(
            f"Persisted intrinsic calibration for port {port}: "
            f"in_sample={output.report.in_sample_rmse:.3f}px, "
            f"out_of_sample={output.report.out_of_sample_rmse:.3f}px"
        )
        self.push_camera_data(port)

    def get_intrinsic_report(self, port: int) -> IntrinsicCalibrationReport | None:
        """Get cached intrinsic calibration report for a port."""
        return self._intrinsic_reports.get(port)

    def get_intrinsic_points(self, port: int) -> list[tuple[int, PointPacket]] | None:
        """Get cached collected points for a port (session-only)."""
        return self._intrinsic_points.get(port)

    # -------------------------------------------------------------------------
    # PointDataBundle API (new system, parallel to CaptureVolume)
    # -------------------------------------------------------------------------

    @property
    def point_data_bundle(self) -> PointDataBundle | None:
        """Get the current PointDataBundle for extrinsic calibration.

        Loading priority:
        1. Return cached bundle if available
        2. Try to load from PointDataBundleRepository
        3. Return None if no data available

        Note: This does NOT fall back to CaptureVolume. The two systems are
        independent during migration. Use legacy CaptureVolume methods if you
        need data from the old system.
        """
        if self._point_data_bundle is not None:
            return self._point_data_bundle

        # Try loading from bundle repository
        if self.bundle_repository.camera_array_path.exists():
            try:
                self._point_data_bundle = self.bundle_repository.load()
                logger.info("Loaded PointDataBundle from repository")
                return self._point_data_bundle
            except PersistenceError as e:
                logger.warning(f"Bundle repository exists but load failed: {e}")

        return None

    def update_bundle(self, bundle: PointDataBundle) -> None:
        """Update the in-memory bundle, emit signal, and persist in background.

        Pattern: Update immediately → emit signal → persist asynchronously.
        This ensures UI responsiveness while maintaining durability.

        Args:
            bundle: The new PointDataBundle to store
        """
        self._point_data_bundle = bundle
        self.bundle_updated.emit()

        # Capture for closure (background worker)
        bundle_to_save = bundle

        def worker(_token, _handle):
            try:
                self.bundle_repository.save(bundle_to_save)
                logger.info("PointDataBundle persisted to disk")
            except PersistenceError as e:
                # Log prominently - user's changes may be lost on restart
                logger.error(f"Failed to persist PointDataBundle: {e}")

        self.task_manager.submit(worker, name="save_point_data_bundle")

    def rotate_calibration_bundle(self, axis: Literal["x", "y", "z"], angle_degrees: float) -> None:
        """Rotate the calibration bundle and persist.

        The bundle's rotate() method returns a new immutable bundle with transformed
        world points and camera extrinsics. We update and persist via update_bundle().
        """
        bundle = self.point_data_bundle
        if bundle is None:
            logger.warning("Cannot rotate: no calibration bundle loaded")
            return
        new_bundle = bundle.rotate(axis, angle_degrees)
        self.update_bundle(new_bundle)

    def set_calibration_bundle_origin(self, sync_index: int) -> None:
        """Set world origin to board position at sync_index and persist.

        Uses the charuco board detected at the given sync_index to define
        a new coordinate frame, transforming all points and cameras accordingly.
        """
        bundle = self.point_data_bundle
        if bundle is None:
            logger.warning("Cannot set origin: no calibration bundle loaded")
            return
        new_bundle = bundle.align_to_object(sync_index)
        self.update_bundle(new_bundle)

    # -------------------------------------------------------------------------
    # CaptureVolume API (legacy system - do not mix with PointDataBundle)
    # -------------------------------------------------------------------------

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
        stage = metadata.get("stage")
        if stage is not None:
            self.capture_volume.stage = stage
        origin_sync_index = metadata.get("origin_sync_index")
        if origin_sync_index is not None:
            self.capture_volume.origin_sync_index = origin_sync_index

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

            # Save camera array (shared by both systems)
            self.camera_repository.save(self.camera_array)

            # Save as PointDataBundle (new system)
            # We have image_points and world_points from earlier in this workflow
            bundle = PointDataBundle(self.camera_array, image_points, world_points)
            self.bundle_repository.save(bundle)
            self._point_data_bundle = bundle
            logger.info("Saved PointDataBundle for extrinsic calibration")

            # Also save CaptureVolume (legacy, can remove once migration complete)
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
        handle.failed.connect(lambda *_: self.post_processing_complete.emit())
        return handle

    def rotate_capture_volume(self, direction: str):
        """Rotate capture volume and persist in background thread."""
        assert self.capture_volume is not None
        self.capture_volume.rotate(direction)
        self.capture_volume_shifted.emit()

        capture_volume = self.capture_volume  # capture for closure

        def worker(_token, _handle):
            self.camera_repository.save(self.camera_array)
            self.capture_volume_repository.save_capture_volume(capture_volume)

        self.task_manager.submit(worker, name="rotate_capture_volume")

    def set_capture_volume_origin_to_board(self, origin_index):
        """Set world origin and persist in background thread."""
        assert self.capture_volume is not None
        self.capture_volume.set_origin_to_board(origin_index, self.charuco)
        self.capture_volume_shifted.emit()

        capture_volume = self.capture_volume  # capture for closure

        def worker(_token, _handle):
            self.camera_repository.save(self.camera_array)
            self.capture_volume_repository.save_capture_volume(capture_volume)

        self.task_manager.submit(worker, name="set_capture_volume_origin")
