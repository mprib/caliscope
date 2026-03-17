import logging
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
from typing import Literal

import cv2
from PySide6.QtCore import QObject, QFileSystemWatcher, Signal

from caliscope.task_manager import TaskHandle, TaskManager

from caliscope.core.charuco import Charuco
from caliscope.core.chessboard import Chessboard
from caliscope.core.aruco_target import ArucoTarget
from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.calibrate_intrinsics import IntrinsicCalibrationOutput, IntrinsicCalibrationReport
from caliscope.repositories import (
    CameraArrayRepository,
    CalibrationTargetsRepository,
    ProjectSettingsRepository,
)
from caliscope.repositories.capture_volume_repository import CaptureVolumeRepository
from caliscope.repositories.calibration_targets_repository import (
    IntrinsicTargetType,
    ExtrinsicTargetType,
    TargetRouting,
)
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.workflow_status import WorkflowStatus
from caliscope.persistence import PersistenceError
from caliscope.repositories.intrinsic_report_repository import IntrinsicReportRepository
from caliscope.reconstruction.reconstructor import Reconstructor
from caliscope.recording import read_video_properties
from caliscope.core.point_data import ImagePoints
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.trackers.chessboard_tracker import ChessboardTracker
from caliscope.trackers.aruco_tracker import ArucoTracker
from caliscope.workspace_guide import WorkspaceGuide
from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    ExtrinsicCalibrationPresenter,
)
from caliscope.gui.presenters.intrinsic_calibration_presenter import IntrinsicCalibrationPresenter
from caliscope.gui.presenters.multi_camera_processing_presenter import MultiCameraProcessingPresenter
from caliscope.gui.presenters.reconstruction_presenter import ReconstructionPresenter
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

    new_camera_data = Signal(int, OrderedDict)  # cam_id, camera_display_dictionary
    intrinsic_target_changed = Signal()  # Emitted when intrinsic target config is updated
    extrinsic_target_changed = Signal()  # Emitted when extrinsic target config is updated
    capture_volume_updated = Signal()  # Immediate: in-memory state changed, use for UI refresh
    status_changed = Signal()  # Deferred: fires after filesystem operations complete

    def __init__(self, workspace_dir: Path):
        super().__init__()
        self.workspace = workspace_dir

        # Initialize repositories with explicit file paths
        self.settings_repository = ProjectSettingsRepository(workspace_dir / "project_settings.toml")
        self.camera_repository = CameraArrayRepository(workspace_dir / "camera_array.toml")
        self.targets_repository = CalibrationTargetsRepository(workspace_dir / "calibration" / "targets")
        self.intrinsic_report_repository = IntrinsicReportRepository(
            workspace_dir / "calibration" / "intrinsic" / "reports"
        )

        # CaptureVolume (extrinsic calibration system)
        # Capture volume lives as a sibling to the tracker extraction directory.
        # Extraction writes to .../ARUCO/image_points.csv, capture volume saves to .../capture_volume/.
        self.capture_volume_repository = CaptureVolumeRepository(
            workspace_dir / "calibration" / "extrinsic" / "capture_volume"
        )
        self._capture_volume: CaptureVolume | None = None

        # Initialize project files if they don't exist
        self._initialize_project_files()

        # streams will be used to play back recorded video with tracked markers to select frames
        self.camera_array = CameraArray({})  # empty camera array at init

        logger.info("Building workspace guide")
        self.workspace_guide = WorkspaceGuide(self.workspace)
        self.workspace_guide.intrinsic_dir.mkdir(exist_ok=True, parents=True)
        self.workspace_guide.extrinsic_dir.mkdir(exist_ok=True, parents=True)
        self.workspace_guide.recording_dir.mkdir(exist_ok=True, parents=True)

        # Watch calibration directories for file changes
        self._setup_filesystem_watcher()

        # Centralized task management for background operations
        self.task_manager = TaskManager(parent=self)

        # Created during process_recordings; lives for the duration of that task
        self.reconstructor: Reconstructor | None = None

        # In-memory cache of intrinsic calibration data (by cam_id)
        # These enable overlay restoration when switching between cameras
        self._intrinsic_reports: dict[int, IntrinsicCalibrationReport] = {}
        # Session-only cache of collected points for overlay rendering
        # Not persisted to disk - lost on app restart
        self._intrinsic_points: dict[int, list[tuple[int, PointPacket]]] = {}

        # Global intrinsic calibration settings
        self._intrinsic_frame_skip: int = 5

    def _setup_filesystem_watcher(self) -> None:
        """Watch calibration directories for file changes."""
        self._watcher = QFileSystemWatcher(parent=self)

        dirs_to_watch = [
            self.workspace_guide.intrinsic_dir,
            self.workspace_guide.extrinsic_dir,
            self.workspace_guide.recording_dir,
        ]

        for dir_path in dirs_to_watch:
            if dir_path.exists():
                self._watcher.addPath(str(dir_path))
                logger.debug(f"Watching directory: {dir_path}")

        self._watcher.directoryChanged.connect(self._on_directory_changed)

    def _on_directory_changed(self, path: str) -> None:
        """Handle filesystem change in watched directory."""
        logger.info(f"Directory changed: {path}")
        self.status_changed.emit()

    @property
    def camera_count(self) -> int:
        """Derived camera count from extrinsic directory (source of truth)."""
        return self.workspace_guide.get_camera_count()

    @property
    def cam_ids(self) -> list[int]:
        """Authoritative list of camera IDs from extrinsic directory."""
        return self.workspace_guide.get_cam_ids()

    @property
    def extrinsic_image_points_path(self) -> Path:
        """Path to 2D observations from extrinsic calibration extraction."""
        tracker_name = self.targets_repository.get_extrinsic_tracker_name()
        return self.workspace_guide.extrinsic_dir / tracker_name / "image_points.csv"

    @property
    def cameras_tab_enabled(self) -> bool:
        """Whether Cameras tab should be enabled.

        Requires: intrinsic videos exist for all cameras in the extrinsic set.
        """
        return self.workspace_guide.all_instrinsic_mp4s_available()

    @property
    def multi_camera_tab_enabled(self) -> bool:
        """Whether Multi-Camera tab should be enabled.

        Requires: extrinsic videos exist AND all intrinsics calibrated.
        """
        return self.workspace_guide.all_extrinsic_mp4s_available() and self.camera_array.all_intrinsics_calibrated()

    @property
    def capture_volume_tab_enabled(self) -> bool:
        """Whether Capture Volume tab should be enabled.

        Requires: 2D extraction complete AND intrinsics calibrated.
        This is where extrinsic calibration happens.
        """
        extraction_complete = self.extrinsic_image_points_path.exists()
        return extraction_complete and self.camera_array.all_intrinsics_calibrated()

    @property
    def reconstruction_tab_enabled(self) -> bool:
        """Whether Reconstruction tab should be enabled.

        Requires: extrinsic calibration complete AND recordings available.
        """
        return self.capture_volume_tab_enabled and self.recordings_available()

    def _initialize_project_files(self):
        """Create default project files if they don't exist."""
        logger.info("Checking for existing project files...")

        # Project settings (always create/update to ensure required fields exist)
        if not self.settings_repository.path.exists():
            logger.info("Creating default project settings")
            self.settings_repository.save(
                {
                    "creation_date": datetime.now().isoformat(),
                    "save_tracked_points_video": True,
                    "fps_sync_stream_processing": 100,
                }
            )

        # Calibration targets (creates all default target configs + routing)
        self.targets_repository.initialize_defaults()

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
            # Load camera array if intrinsic videos exist
            if self.cameras_tab_enabled:
                logger.info("Loading camera array (intrinsic videos available)")
                self.load_camera_array()
            else:
                logger.info("Skipping camera array load (no intrinsic videos)")

            # Load capture volume if extrinsic calibration complete
            if self.capture_volume_tab_enabled:
                logger.info("Extrinsic calibration available (loaded via capture_volume property)")
            else:
                logger.info("Skipping capture volume load (not calibrated)")

        handle = self.task_manager.submit(worker, name="load_workspace", auto_start=False)
        handle.completed.connect(lambda _: self.status_changed.emit())
        self.task_manager.start_task(handle.task_id)
        return handle

    def all_instrinsic_mp4s_available(self) -> bool:
        """Check if all intrinsic calibration videos are present."""
        return self.workspace_guide.all_instrinsic_mp4s_available()

    def all_extrinsic_mp4s_available(self) -> bool:
        """Check if all extrinsic calibration videos are present."""
        return self.workspace_guide.all_extrinsic_mp4s_available()

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

        # Check for calibration data in CaptureVolume system
        point_estimates_good = self.capture_volume_repository.camera_array_path.exists()
        logger.info(f"Point estimates available: {point_estimates_good}")

        all_data_available = self.workspace_guide.all_extrinsic_mp4s_available()
        logger.info(f"All underlying data available: {all_data_available}")

        return cameras_good and point_estimates_good and all_data_available

    def recordings_available(self) -> bool:
        """Check if any valid recording directories exist."""
        return len(self.workspace_guide.valid_recording_dirs()) > 0

    def get_workflow_status(self) -> WorkflowStatus:
        """Compute current workflow status from ground truth.

        This method queries the filesystem and domain objects to build
        a status snapshot. Called by the Project tab whenever it refreshes.
        """
        camera_count = self.camera_count  # Now a property
        expected_cam_ids = set(self.cam_ids) if self.cam_ids else set()

        # Intrinsic video availability
        intrinsic_cam_ids = self.workspace_guide.get_cam_ids_in_dir(self.workspace_guide.intrinsic_dir)
        intrinsic_missing = sorted(expected_cam_ids - set(intrinsic_cam_ids))

        # Extrinsic video availability
        extrinsic_cam_ids = self.workspace_guide.get_cam_ids_in_dir(self.workspace_guide.extrinsic_dir)
        extrinsic_missing = sorted(expected_cam_ids - set(extrinsic_cam_ids))

        # Cameras needing intrinsic calibration
        cameras_needing = [cam_id for cam_id, cam in self.camera_array.cameras.items() if cam.matrix is None]

        # 2D extraction complete check
        extraction_complete = self.extrinsic_image_points_path.exists()

        return WorkflowStatus(
            camera_count=camera_count,
            charuco_configured=True,
            intrinsic_videos_available=len(intrinsic_missing) == 0,
            intrinsic_videos_missing=intrinsic_missing,
            intrinsic_calibration_complete=self.camera_array.all_intrinsics_calibrated(),
            cameras_needing_calibration=cameras_needing,
            extrinsic_videos_available=len(extrinsic_missing) == 0,
            extrinsic_videos_missing=extrinsic_missing,
            extrinsic_2d_extraction_complete=extraction_complete,
            extrinsic_calibration_complete=self.all_extrinsics_estimated(),
            recordings_available=self.recordings_available(),
            recording_names=self.workspace_guide.valid_recording_dirs(),
        )

    def update_intrinsic_target_type(self, target_type: IntrinsicTargetType) -> None:
        """Update which target type is used for intrinsic calibration."""
        routing = self.targets_repository.get_routing()
        new_routing = TargetRouting(
            intrinsic_target_type=target_type,
            extrinsic_target_type=routing.extrinsic_target_type,
            extrinsic_charuco_same_as_intrinsic=routing.extrinsic_charuco_same_as_intrinsic,
        )
        self.targets_repository.save_routing(new_routing)
        self.intrinsic_target_changed.emit()

    def update_extrinsic_target_type(self, target_type: ExtrinsicTargetType) -> None:
        """Update which target type is used for extrinsic calibration."""
        routing = self.targets_repository.get_routing()
        new_routing = TargetRouting(
            intrinsic_target_type=routing.intrinsic_target_type,
            extrinsic_target_type=target_type,
            extrinsic_charuco_same_as_intrinsic=routing.extrinsic_charuco_same_as_intrinsic,
        )
        self.targets_repository.save_routing(new_routing)
        self.extrinsic_target_changed.emit()

    def update_intrinsic_charuco(self, charuco: Charuco) -> None:
        """Persist intrinsic charuco config and notify consumers."""
        self.targets_repository.save_intrinsic_charuco(charuco)
        self.intrinsic_target_changed.emit()
        # If extrinsic shares intrinsic charuco, extrinsic consumers need to know too
        routing = self.targets_repository.get_routing()
        if routing.extrinsic_target_type == "charuco" and routing.extrinsic_charuco_same_as_intrinsic:
            self.extrinsic_target_changed.emit()

    def update_intrinsic_chessboard(self, chessboard: Chessboard) -> None:
        """Persist intrinsic chessboard config and notify consumers."""
        self.targets_repository.save_chessboard(chessboard)
        self.intrinsic_target_changed.emit()

    def update_extrinsic_charuco(self, charuco: Charuco) -> None:
        """Persist extrinsic-specific charuco config and notify consumers."""
        self.targets_repository.save_extrinsic_charuco(charuco)
        self.extrinsic_target_changed.emit()

    def update_extrinsic_aruco_target(self, target: ArucoTarget) -> None:
        """Persist extrinsic ArUco target config and notify consumers."""
        self.targets_repository.save_aruco_target(target)
        self.extrinsic_target_changed.emit()

    def set_extrinsic_charuco_same_as_intrinsic(self, same: bool) -> None:
        """Toggle whether extrinsic charuco shares intrinsic config."""
        self.targets_repository.set_extrinsic_charuco_same_as_intrinsic(same)
        self.extrinsic_target_changed.emit()

    def create_intrinsic_tracker(self) -> ChessboardTracker | CharucoTracker:
        """Create tracker for intrinsic calibration based on current target type."""
        target_type = self.targets_repository.intrinsic_target_type
        if target_type == "chessboard":
            chessboard = self.targets_repository.load_chessboard()
            return ChessboardTracker(chessboard)
        else:  # "charuco"
            charuco = self.targets_repository.load_intrinsic_charuco()
            return CharucoTracker(charuco)

    def create_extrinsic_tracker(self) -> ArucoTracker | CharucoTracker:
        """Create tracker for extrinsic calibration based on current target type."""
        target_type = self.targets_repository.extrinsic_target_type
        if target_type == "aruco":
            if not self.targets_repository.aruco_target_exists():
                # Create default if missing (backward compat with first-time setup)
                default_target = ArucoTarget.single_marker(
                    marker_id=0,
                    marker_size_m=0.05,
                    dictionary=cv2.aruco.DICT_4X4_100,
                )
                self.targets_repository.save_aruco_target(default_target)
            target = self.targets_repository.load_aruco_target()
            return ArucoTracker(
                dictionary=target.dictionary,
                aruco_target=target,
            )
        else:  # "charuco"
            charuco = self.targets_repository.load_extrinsic_charuco()
            return CharucoTracker(charuco)

    def load_camera_array(self):
        """
        Load camera array from persistence and detect new cameras from video files.

        Any cameras discovered in the intrinsic directory that aren't in the
        saved array will be added automatically. Also loads any persisted
        intrinsic calibration reports for overlay restoration.
        """
        self.camera_array = self.camera_repository.load()

        # double check that no new camera associated files have been placed in the intrinsic calibration folder
        all_cam_ids = self.workspace_guide.get_cam_ids_in_dir(self.workspace_guide.intrinsic_dir)

        for cam_id in all_cam_ids:
            if cam_id not in self.camera_array.cameras:
                self._add_camera_from_source(cam_id)

        # Load any persisted intrinsic reports for overlay restoration
        self._intrinsic_reports = self.intrinsic_report_repository.load_all()
        if self._intrinsic_reports:
            logger.info(f"Loaded intrinsic reports for cam_ids: {list(self._intrinsic_reports.keys())}")

    def _add_camera_from_source(self, cam_id: int):
        """
        Add a new camera discovered from video file in intrinsic directory.

        File will be transferred to workspace/calibration/intrinsic/cam_{cam_id}.mp4
        in keeping with project layout.
        """
        # copy source over to standard workspace structure
        target_mp4_path = Path(self.workspace_guide.intrinsic_dir, f"cam_{cam_id}.mp4")
        video_properties = read_video_properties(target_mp4_path)
        size = video_properties["size"]
        new_cam_data = CameraData(
            cam_id=cam_id,
            size=size,
        )
        self.camera_array.cameras[cam_id] = new_cam_data
        self.camera_repository.save(self.camera_array)

    def push_camera_data(self, cam_id):
        """Emit signal with updated camera display data."""
        logger.info(f"Pushing camera data for cam_id {cam_id}")
        camera_display_data = self.camera_array.cameras[cam_id].get_display_data()
        logger.info(f"camera display data is {camera_display_data}")
        self.new_camera_data.emit(cam_id, camera_display_data)

    def create_intrinsic_presenter(self, cam_id: int) -> IntrinsicCalibrationPresenter:
        """Create presenter for intrinsic calibration of a single camera.

        Factory method that assembles the presenter with all required dependencies.
        The caller is responsible for connecting signals and managing presenter lifecycle.

        If a previous calibration exists (report in cache), it's passed to the presenter
        for overlay restoration. Collected points are only available during the session
        (not after app restart).

        Raises:
            ValueError: If cam_id is not in camera_array or intrinsic video doesn't exist.
        """
        if cam_id not in self.camera_array.cameras:
            raise ValueError(f"No camera data for cam_id {cam_id}")

        camera = self.camera_array.cameras[cam_id]
        video_path = self.workspace_guide.intrinsic_dir / f"cam_{cam_id}.mp4"

        if not video_path.exists():
            raise ValueError(f"No intrinsic video for cam_id {cam_id}")

        # Get cached data for overlay restoration
        report = self._intrinsic_reports.get(cam_id)
        collected_points = self._intrinsic_points.get(cam_id)

        return IntrinsicCalibrationPresenter(
            camera=camera,
            video_path=video_path,
            tracker=self.create_intrinsic_tracker(),
            task_manager=self.task_manager,
            restored_report=report,
            restored_points=collected_points,
            frame_skip=self._intrinsic_frame_skip,
        )

    @property
    def intrinsic_frame_skip(self) -> int:
        """Current frame skip value for intrinsic calibration."""
        return self._intrinsic_frame_skip

    def set_intrinsic_frame_skip(
        self, value: int, presenters: dict[int, IntrinsicCalibrationPresenter] | None = None
    ) -> None:
        """Set global frame skip and propagate to all active presenters.

        Args:
            value: Process every Nth frame (minimum 1).
            presenters: Active presenter pool to propagate to. Passed by CamerasTabWidget
                since the coordinator doesn't own the presenter pool.
        """
        self._intrinsic_frame_skip = max(1, value)
        if presenters is not None:
            for presenter in presenters.values():
                presenter.set_frame_skip(self._intrinsic_frame_skip)

    def create_reconstruction_presenter(self) -> ReconstructionPresenter:
        """Create presenter for reconstruction (post-processing) workflow.

        Factory method that assembles the presenter with all required dependencies.
        The caller is responsible for connecting signals and managing presenter lifecycle.

        Returns:
            ReconstructionPresenter configured with workspace and camera array.
        """
        return ReconstructionPresenter(
            workspace_dir=self.workspace,
            camera_array=self.camera_array,
            task_manager=self.task_manager,
            project_settings=self.settings_repository,
        )

    def create_multi_camera_presenter(self) -> MultiCameraProcessingPresenter:
        """Create presenter for multi-camera synchronized video processing.

        Factory method that assembles the presenter with all required dependencies.
        The caller is responsible for:
        - Calling set_recording_dir() and set_cameras() to configure
        - Managing presenter lifecycle (cleanup on tab close)

        Returns:
            MultiCameraProcessingPresenter configured with task_manager and tracker.
        """
        presenter = MultiCameraProcessingPresenter(
            task_manager=self.task_manager,
            tracker=self.create_extrinsic_tracker(),
        )

        # Wire signal directly - no passthrough needed
        presenter.processing_complete.connect(lambda ip, _cr, t: self.persist_extrinsic_image_points(ip, t.name))

        return presenter

    def create_extrinsic_calibration_presenter(self) -> ExtrinsicCalibrationPresenter:
        """Create presenter for extrinsic calibration workflow.

        Factory method that assembles the presenter with all required dependencies.
        The presenter handles bootstrap triangulation, bundle adjustment, and
        coordinate frame transformations.

        If a capture volume already exists (from a previous session), it's passed
        to the presenter so the UI starts in the CALIBRATED state with visualization.

        The caller is responsible for:
        - Managing presenter lifecycle (cleanup on tab close)

        Returns:
            ExtrinsicCalibrationPresenter configured with camera_array, image_points, etc.
        """
        # ImagePoints path from multi-camera processing (Phase 3 output)
        image_points_path = self.extrinsic_image_points_path

        # Check for existing calibration (restores state on project reopen)
        existing_capture_volume = self.capture_volume

        presenter = ExtrinsicCalibrationPresenter(
            task_manager=self.task_manager,
            camera_array=self.camera_array,
            image_points_path=image_points_path,
            existing_capture_volume=existing_capture_volume,
            project_settings=self.settings_repository,
        )

        # Wire signal directly - no passthrough needed
        presenter.capture_volume_changed.connect(self.update_capture_volume)

        return presenter

    def persist_extrinsic_image_points(self, image_points: ImagePoints, tracker_name: str) -> None:
        """Persist 2D image points from multi-camera processing.

        Saves ImagePoints to the extrinsic calibration directory for use by
        the Extrinsic Calibration tab. The full CaptureVolume (with WorldPoints)
        is created later after bootstrapping and triangulation.

        Args:
            image_points: 2D observations from synchronized video processing
            tracker_name: Tracker name for subfolder (e.g., "CHARUCO")
        """
        # Ensure the tracker directory exists
        tracker_dir = self.workspace_guide.extrinsic_dir / tracker_name
        tracker_dir.mkdir(parents=True, exist_ok=True)

        output_path = tracker_dir / "image_points.csv"
        image_points.to_csv(output_path)

        logger.info(f"Persisted extrinsic image points: {len(image_points.df)} observations to {output_path}")
        self.status_changed.emit()

    def persist_camera_rotation(self, cam_id: int, rotation_count: int) -> None:
        """Persist updated rotation for a camera.

        Called when user adjusts camera orientation in the multi-camera processing
        view. Updates both the in-memory camera array and disk persistence.

        Args:
            cam_id: Camera ID to update
            rotation_count: Rotation in 90° increments (0-3)
        """
        if cam_id not in self.camera_array.cameras:
            logger.warning(f"Cannot persist rotation: cam_id {cam_id} not in camera_array")
            return

        self.camera_array.cameras[cam_id].rotation_count = rotation_count
        self.camera_repository.save(self.camera_array)
        self.push_camera_data(cam_id)
        logger.debug(f"Persisted camera rotation: cam_id {cam_id} -> {rotation_count * 90}°")

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
        cam_id = output.camera.cam_id

        # Update camera in array and save
        self.camera_array.cameras[cam_id] = output.camera
        self.camera_repository.save(self.camera_array)

        # Cache report for overlay restoration and save to disk
        self._intrinsic_reports[cam_id] = output.report
        self.intrinsic_report_repository.save(cam_id, output.report)

        # Cache collected points for session-only overlay restoration
        if collected_points is not None:
            self._intrinsic_points[cam_id] = collected_points

        logger.info(f"Persisted intrinsic calibration for cam_id {cam_id}: rmse={output.report.rmse:.3f}px")
        self.push_camera_data(cam_id)
        self.status_changed.emit()

    def get_intrinsic_report(self, cam_id: int) -> IntrinsicCalibrationReport | None:
        """Get cached intrinsic calibration report for a camera."""
        return self._intrinsic_reports.get(cam_id)

    def get_intrinsic_points(self, cam_id: int) -> list[tuple[int, PointPacket]] | None:
        """Get cached collected points for a camera (session-only)."""
        return self._intrinsic_points.get(cam_id)

    # -------------------------------------------------------------------------
    # CaptureVolume API
    # -------------------------------------------------------------------------

    @property
    def capture_volume(self) -> CaptureVolume | None:
        """Get the current CaptureVolume for extrinsic calibration.

        Loading priority:
        1. Return cached capture volume if available
        2. Try to load from CaptureVolumeRepository
        3. Return None if no data available
        """
        if self._capture_volume is not None:
            return self._capture_volume

        # Try loading from capture volume repository
        if self.capture_volume_repository.camera_array_path.exists():
            try:
                self._capture_volume = self.capture_volume_repository.load()
                logger.info("Loaded CaptureVolume from repository")
                return self._capture_volume
            except PersistenceError as e:
                logger.warning(f"Bundle repository exists but load failed: {e}")

        return None

    def update_capture_volume(self, capture_volume: CaptureVolume) -> None:
        """Update the in-memory capture volume and persist in background.

        Emits capture_volume_updated immediately (for UI refresh using in-memory state).
        Emits status_changed after save completes (for filesystem-based status checks).

        Also updates the main camera_array so that on restart, the calibrated
        extrinsics are available (enables tab and correct state detection).

        Args:
            capture_volume: The new CaptureVolume to store
        """
        self._capture_volume = capture_volume
        self.camera_array = capture_volume.camera_array  # Keep main camera_array in sync
        self.capture_volume_updated.emit()  # Immediate - consumers use in-memory state

        # Capture for closure (background worker)
        capture_volume_to_save = capture_volume
        camera_repo = self.camera_repository
        aniposelib_path = self.workspace / "camera_array_aniposelib.toml"

        def worker(_token, _handle):
            try:
                self.capture_volume_repository.save(capture_volume_to_save)
                logger.info("CaptureVolume persisted to disk")
                # Also save camera_array to main repo for restart detection
                camera_repo.save(capture_volume_to_save.camera_array)
                logger.info("Camera array with extrinsics persisted")
                # Export aniposelib-compatible format to workspace root for downstream tools
                capture_volume_to_save.camera_array.to_aniposelib_toml(aniposelib_path)
                logger.info("Aniposelib-compatible camera array exported")
            except PersistenceError as e:
                # Log prominently - user's changes may be lost on restart
                logger.error(f"Failed to persist CaptureVolume: {e}")

        handle = self.task_manager.submit(worker, name="save_capture_volume", auto_start=False)
        handle.completed.connect(lambda _: self.status_changed.emit())  # Post-save
        self.task_manager.start_task(handle.task_id)

    def rotate_capture_volume(self, axis: Literal["x", "y", "z"], angle_degrees: float) -> None:
        """Rotate the capture volume and persist.

        The CaptureVolume.rotate() method returns a new immutable instance with transformed
        world points and camera extrinsics. We update and persist via update_capture_volume().
        """
        capture_volume = self.capture_volume
        if capture_volume is None:
            logger.warning("Cannot rotate: no capture volume loaded")
            return
        new_capture_volume = capture_volume.rotate(axis, angle_degrees)
        self.update_capture_volume(new_capture_volume)

    def set_capture_volume_origin(self, sync_index: int) -> None:
        """Set world origin to board position at sync_index and persist.

        Uses the charuco board detected at the given sync_index to define
        a new coordinate frame, transforming all points and cameras accordingly.
        """
        capture_volume = self.capture_volume
        if capture_volume is None:
            logger.warning("Cannot set origin: no capture volume loaded")
            return
        new_capture_volume = capture_volume.align_to_object(sync_index)
        self.update_capture_volume(new_capture_volume)

    def process_recordings(self, recording_path: Path, tracker_name: str) -> TaskHandle:
        """
        Initiate post-processing of recorded video in worker thread.

        Args:
            recording_path: Directory containing synchronized video recordings
            tracker_name: Tracker name to use for landmark detection

        Returns:
            TaskHandle for connecting completion callbacks.
        """

        def worker(token, handle):
            logger.info(f"Beginning to process video files at {recording_path}")
            logger.info(f"Creating reconstructor for {recording_path}")
            self.reconstructor = Reconstructor(self.camera_array, recording_path, tracker_name)

            # Get processing settings from project configuration
            include_video = self.settings_repository.get_save_tracked_points_video()
            fps_target = self.settings_repository.get_fps_sync_stream_processing()

            # Pass token for cancellation support and handle for progress reporting
            if not self.reconstructor.create_xy(
                include_video=include_video, fps_target=fps_target, token=token, handle=handle
            ):
                return  # Cancelled

            # Stage 2 progress (80-100%)
            handle.report_progress(85, "Stage 2: Triangulating 3D points")
            self.reconstructor.create_xyz()
            handle.report_progress(100, "Complete")

        handle = self.task_manager.submit(worker, name="process_recordings", auto_start=False)
        self.task_manager.start_task(handle.task_id)
        return handle

    def cleanup(self) -> None:
        """Shutdown all background operations.

        Should be called when the application is closing to ensure threads
        are properly terminated. The TaskManager handles its own thread
        pool shutdown with configurable timeout.
        """
        logger.info("WorkspaceCoordinator cleanup initiated")
        self.task_manager.shutdown(timeout_ms=5000)
        logger.info("WorkspaceCoordinator cleanup complete")
