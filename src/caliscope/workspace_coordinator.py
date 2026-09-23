import logging
from pathlib import Path
from datetime import datetime
from typing import NamedTuple

import cv2
from PySide6.QtCore import QObject, QFileSystemWatcher, Qt, QTimer, Signal

from caliscope.task_manager import TaskHandle, TaskManager

from caliscope.core.charuco import Charuco
from caliscope.core.chessboard import Chessboard
from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet
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
from caliscope.core.constraints import ConstraintSet
from caliscope.core.workflow_status import WorkflowStatus
from caliscope.persistence import PersistenceError
from caliscope.repositories.intrinsic_report_repository import IntrinsicReportRepository
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

RECORDING_POLL_INTERVAL_MS = 2000


class _SessionSnapshot(NamedTuple):
    """Direct-child names and canonical video (mtime_ns, size) of one session."""

    names: frozenset[str]
    videos: dict[Path, tuple[int, int]]


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

    intrinsic_target_changed = Signal()  # Emitted when intrinsic target config is updated
    extrinsic_target_changed = Signal()  # Emitted when extrinsic target config is updated
    status_changed = Signal()  # Deferred: fires after filesystem operations complete
    recording_directory_changed = Signal(Path)  # Recording root or immediate session changed
    recording_video_changed = Signal(Path)  # Canonical recording video changed
    calibration_changed = Signal()  # Camera array changed, use for reconstruction refresh

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

        # In-memory cache of intrinsic calibration data (by cam_id)
        # These enable overlay restoration when switching between cameras
        self._intrinsic_reports: dict[int, IntrinsicCalibrationReport] = {}
        # Session-only cache of collected points for overlay rendering
        # Not persisted to disk - lost on app restart
        self._intrinsic_points: dict[int, list[tuple[int, PointPacket]]] = {}

        # Global intrinsic calibration settings
        self._intrinsic_frame_skip: int = 5

    def _setup_filesystem_watcher(self) -> None:
        """Watch calibration and recording roots, and poll recording sessions.

        Only directories the workspace keeps are watched. Session folders and
        their videos are polled instead: on Windows, Qt's watcher can hang in
        removePaths() after a watched folder nested in another watched folder
        is deleted.
        """
        self._watcher = QFileSystemWatcher(parent=self)

        dirs_to_watch = [
            self.workspace_guide.intrinsic_dir,
            self.workspace_guide.extrinsic_dir,
            self.workspace_guide.recording_dir,
        ]

        for dir_path in dirs_to_watch:
            if dir_path.exists():
                self._watcher.addPath(str(dir_path.resolve()))
                logger.debug(f"Watching directory: {dir_path}")

        self._watcher.directoryChanged.connect(self._on_directory_changed)

        self._recording_snapshot = self._snapshot_recordings()
        self._recording_poll_timer = QTimer(self)
        self._recording_poll_timer.setInterval(RECORDING_POLL_INTERVAL_MS)
        self._recording_poll_timer.timeout.connect(self._poll_recordings)
        self._recording_poll_timer.start()

    def _snapshot_recordings(self) -> dict[Path, _SessionSnapshot]:
        """Return each session's entry names and canonical video stats."""
        recording_dir = self.workspace_guide.recording_dir
        snapshot: dict[Path, _SessionSnapshot] = {}
        try:
            session_dirs = [path for path in recording_dir.iterdir() if path.is_dir()]
        except OSError:
            return snapshot

        recording_root = recording_dir.absolute()
        for session_dir in session_dirs:
            session_path = recording_root / session_dir.name
            try:
                names = frozenset(path.name for path in session_dir.iterdir())
                videos: dict[Path, tuple[int, int]] = {}
                for cam_id in self.workspace_guide.get_cam_ids_in_dir(session_dir):
                    # Keyed by the in-session path, so replacing a linked video
                    # with a regular file reads as an edit of the same video.
                    video = session_path / f"cam_{cam_id}.mp4"
                    stat = video.stat()
                    videos[video] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                # The session changed mid-scan. The next poll sees it settled.
                continue
            snapshot[session_path] = _SessionSnapshot(names, videos)
        return snapshot

    def _poll_recordings(self, *, root_changed: bool = False) -> None:
        """Emit recording signals for what changed since the last snapshot.

        root_changed forces a root notice for a watcher event, which also
        covers loose files at the root that no session snapshot records.
        """
        previous = self._recording_snapshot
        current = self._snapshot_recordings()
        self._recording_snapshot = current

        changed = root_changed or previous.keys() != current.keys()
        if changed:
            self.recording_directory_changed.emit(self.workspace_guide.recording_dir.absolute())
        for session_dir in previous.keys() & current.keys():
            before, after = previous[session_dir], current[session_dir]
            if before.names != after.names:
                self.recording_directory_changed.emit(session_dir)
                changed = True
            for video, stats in after.videos.items():
                if video in before.videos and before.videos[video] != stats:
                    self.recording_video_changed.emit(video)
                    changed = True
        if changed:
            self.status_changed.emit()

    def _on_directory_changed(self, path: str) -> None:
        """Handle filesystem change in watched directory."""
        logger.info(f"Directory changed: {path}")
        changed_path = Path(path).absolute()
        if changed_path.resolve() in {
            self.workspace_guide.intrinsic_dir.resolve(),
            self.workspace_guide.extrinsic_dir.resolve(),
        }:
            self._discover_new_cameras()
        if changed_path.resolve() == self.workspace_guide.recording_dir.resolve():
            self._poll_recordings(root_changed=True)
            return
        self.status_changed.emit()

    @property
    def expected_cam_ids(self) -> set[int]:
        """Cameras the workspace expects calibration videos for.

        Extrinsic videos define the camera set. Loaded cameras keep that set
        stable once a video disappears, so a deleted extrinsic file is reported
        as missing instead of shrinking the set and blaming its intrinsic twin.
        """
        return set(self.camera_array.cameras) | set(self.workspace_guide.get_cam_ids())

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

        Requires: extrinsic videos exist AND all cameras have resolution.
        """
        return self.workspace_guide.all_extrinsic_mp4s_available() and self.camera_array.all_cameras_have_resolution()

    @property
    def capture_volume_tab_enabled(self) -> bool:
        """Whether Calibrate tab should be enabled.

        Requires: 2D extraction complete AND all cameras have resolution.
        """
        extraction_complete = self.extrinsic_image_points_path.exists()
        return extraction_complete and self.camera_array.all_cameras_have_resolution()

    @property
    def reconstruction_tab_enabled(self) -> bool:
        """Whether Reconstruction tab should be enabled.

        Requires a calibrated capture volume. Recording availability and file
        feedback are rendered inside the tab.
        """
        return self.capture_volume_repository.camera_array_path.exists()

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

        def worker(token, _handle):
            if token.is_cancelled:
                return

            # Load camera array if any calibration videos exist
            has_intrinsic = self.workspace_guide.all_instrinsic_mp4s_available()
            has_extrinsic = self.workspace_guide.all_extrinsic_mp4s_available()
            if has_intrinsic or has_extrinsic:
                logger.info("Loading camera array (calibration videos available)")
                self.load_camera_array()
            else:
                logger.info("Skipping camera array load (no calibration videos)")

            if token.is_cancelled:
                return

            # Overlay bundle-authoritative camera state if a bundle exists
            if self.capture_volume_repository.camera_array_path.exists():
                bundle = self.capture_volume
                if bundle is not None:
                    for cam_id, bundle_cam in bundle.camera_array.cameras.items():
                        if cam_id in self.camera_array.cameras:
                            self.camera_array.cameras[cam_id] = bundle_cam
                    logger.info("Overlaid bundle-authoritative camera state")

            if token.is_cancelled:
                return

            if self.capture_volume_tab_enabled:
                logger.info("Extrinsic calibration available (loaded via capture_volume property)")
            else:
                logger.info("Skipping capture volume load (not calibrated)")

        handle = self.task_manager.submit(worker, name="load_workspace", auto_start=False)

        def _on_workspace_loaded(_: object) -> None:
            # This runs after the worker has overlaid bundle-authoritative
            # cameras, so observers receive one complete calibration snapshot.
            self.calibration_changed.emit()
            self.status_changed.emit()

        handle.completed.connect(_on_workspace_loaded)
        return handle

    def start_load(self, handle: TaskHandle) -> None:
        self.task_manager.start_task(handle.task_id)

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

    def _reconstruction_cam_ids(self) -> set[int]:
        """Camera IDs required by the camera array used for reconstruction."""
        return set(self.camera_array.cameras)

    def get_workflow_status(self) -> WorkflowStatus:
        """Compute current workflow status from ground truth.

        This method queries the filesystem and domain objects to build
        a status snapshot. Called by the Project tab whenever it refreshes.
        """
        expected_cam_ids = self.expected_cam_ids
        extrinsic_assessment = self.workspace_guide.assess_extrinsic_videos(expected_cam_ids)
        intrinsic_assessment = self.workspace_guide.assess_intrinsic_videos(expected_cam_ids)
        camera_count = len(expected_cam_ids)
        reconstruction_cam_ids = self._reconstruction_cam_ids()
        recording_assessments = self.workspace_guide.assess_recordings(reconstruction_cam_ids)
        ready_recording_names = [
            name for name, assessment in recording_assessments.items() if reconstruction_cam_ids and assessment.is_ready
        ]
        recording_issues = (
            *self.workspace_guide.recording_layout_issues(),
            *(issue for assessment in recording_assessments.values() for issue in assessment.issues),
        )

        # Cameras needing intrinsic calibration
        cameras_needing = [cam_id for cam_id, cam in self.camera_array.cameras.items() if cam.matrix is None]

        # 2D extraction complete check
        extraction_complete = self.extrinsic_image_points_path.exists()

        return WorkflowStatus(
            camera_count=camera_count,
            cam_ids=sorted(expected_cam_ids),
            charuco_configured=True,
            intrinsic_videos_available=(bool(expected_cam_ids) and not intrinsic_assessment.missing_camera_ids),
            intrinsic_videos_missing=list(intrinsic_assessment.missing_camera_ids),
            intrinsic_video_issues=intrinsic_assessment.issues,
            intrinsic_calibration_complete=self.camera_array.all_intrinsics_calibrated(),
            cameras_needing_calibration=cameras_needing,
            cameras_have_resolution=self.camera_array.all_cameras_have_resolution(),
            extrinsic_videos_available=(bool(expected_cam_ids) and not extrinsic_assessment.missing_camera_ids),
            extrinsic_videos_missing=list(extrinsic_assessment.missing_camera_ids),
            extrinsic_video_issues=extrinsic_assessment.issues,
            extrinsic_2d_extraction_complete=extraction_complete,
            extrinsic_calibration_complete=self.all_extrinsics_estimated(),
            recording_names=list(recording_assessments),
            ready_recording_names=ready_recording_names,
            recording_issues=recording_issues,
        )

    def _discover_new_cameras(self) -> None:
        """Add canonical camera files that appeared after workspace loading."""
        # An empty in-memory array has nothing unsaved to lose (bundle overlays
        # only exist once cameras are loaded), so start from the persisted file
        # rather than overwriting it with a freshly discovered subset.
        camera_array_changed = False
        if not self.camera_array.cameras:
            self.camera_array = self.camera_repository.load()
            camera_array_changed = bool(self.camera_array.cameras)
        if camera_array_changed:
            self.calibration_changed.emit()

        intrinsic_ids = set(self.workspace_guide.get_cam_ids_in_dir(self.workspace_guide.intrinsic_dir))
        extrinsic_ids = set(self.workspace_guide.get_cam_ids_in_dir(self.workspace_guide.extrinsic_dir))
        new_cam_ids = sorted((intrinsic_ids | extrinsic_ids) - set(self.camera_array.cameras))
        if not new_cam_ids:
            return

        # Reading video headers is file I/O; keep it off the GUI thread.
        def worker(_token, _handle) -> dict[int, tuple[int, int]]:
            sizes: dict[int, tuple[int, int]] = {}
            for cam_id in new_cam_ids:
                try:
                    size = self._read_camera_size(cam_id)
                except Exception as error:
                    logger.warning("Could not add camera %s from new video files: %s", cam_id, error)
                    continue
                if size is not None:
                    sizes[cam_id] = size
            return sizes

        handle = self.task_manager.submit(worker, name="Discover cameras", auto_start=False)
        handle.completed.connect(self._on_new_camera_sizes_read, Qt.ConnectionType.QueuedConnection)
        self.task_manager.start_task(handle.task_id)

    def _on_new_camera_sizes_read(self, sizes: dict[int, tuple[int, int]]) -> None:
        """Add cameras whose video headers were read in the background."""
        added = False
        for cam_id, size in sizes.items():
            # A later discovery or load may have added the camera meanwhile.
            if cam_id not in self.camera_array.cameras:
                self.camera_array.cameras[cam_id] = CameraData(cam_id=cam_id, size=size)
                added = True
        if added:
            self.camera_repository.save(self.camera_array)
            self.calibration_changed.emit()
            self.status_changed.emit()

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

    def update_extrinsic_aruco_marker_set(self, marker_set: ArucoMarkerSet) -> None:
        """Persist extrinsic ArUco marker set config and notify consumers."""
        self.targets_repository.save_aruco_marker_set(marker_set)
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
            if not self.targets_repository.aruco_marker_set_exists():
                default_set = ArucoMarkerSet(
                    dictionary=cv2.aruco.DICT_4X4_100,
                    markers={0: ArucoMarker(marker_id=0, size_m=0.05)},
                )
                self.targets_repository.save_aruco_marker_set(default_set)
            marker_set = self.targets_repository.load_aruco_marker_set()
            return ArucoTracker(
                dictionary=marker_set.dictionary,
                marker_set=marker_set,
            )
        else:  # "charuco"
            charuco = self.targets_repository.load_extrinsic_charuco()
            return CharucoTracker(charuco)

    def load_camera_array(self):
        """Load camera array from persistence and detect new cameras from video files.

        Discovers cameras from the union of intrinsic and extrinsic directories.
        Also loads any persisted intrinsic calibration reports for overlay restoration.
        """
        self.camera_array = self.camera_repository.load()

        intrinsic_ids = set(self.workspace_guide.get_cam_ids_in_dir(self.workspace_guide.intrinsic_dir))
        extrinsic_ids = set(self.workspace_guide.get_cam_ids_in_dir(self.workspace_guide.extrinsic_dir))
        all_cam_ids = intrinsic_ids | extrinsic_ids

        for cam_id in sorted(all_cam_ids):
            if cam_id not in self.camera_array.cameras:
                size = self._read_camera_size(cam_id)
                if size is not None:
                    self.camera_array.cameras[cam_id] = CameraData(cam_id=cam_id, size=size)
                    self.camera_repository.save(self.camera_array)

        # Load any persisted intrinsic reports for overlay restoration
        self._intrinsic_reports = self.intrinsic_report_repository.load_all()
        if self._intrinsic_reports:
            logger.info(f"Loaded intrinsic reports for cam_ids: {list(self._intrinsic_reports.keys())}")

    def _read_camera_size(self, cam_id: int) -> tuple[int, int] | None:
        """Read a camera's resolution from its video header.

        Tries intrinsic directory first, falls back to extrinsic. Returns None
        when neither video exists. Raises CalibrationError if both exist with
        different resolutions. Opens video files, so call it off the GUI thread.
        """
        from caliscope.exceptions import CalibrationError

        intrinsic_path = self.workspace_guide.intrinsic_dir / f"cam_{cam_id}.mp4"
        extrinsic_path = self.workspace_guide.extrinsic_dir / f"cam_{cam_id}.mp4"

        if intrinsic_path.exists() and extrinsic_path.exists():
            i_props = read_video_properties(intrinsic_path)
            e_props = read_video_properties(extrinsic_path)
            if i_props["size"] != e_props["size"]:
                raise CalibrationError(
                    f"Resolution mismatch for cam_{cam_id}: intrinsic {i_props['size']} vs extrinsic {e_props['size']}"
                )
            size = i_props["size"]
        elif intrinsic_path.exists():
            size = read_video_properties(intrinsic_path)["size"]
        elif extrinsic_path.exists():
            size = read_video_properties(extrinsic_path)["size"]
        else:
            logger.warning(f"No video found for cam_{cam_id}")
            return None
        return size

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
            workspace_guide=self.workspace_guide,
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
            MultiCameraProcessingPresenter configured with task_manager, tracker,
            and the shared workspace guide.
        """
        presenter = MultiCameraProcessingPresenter(
            task_manager=self.task_manager,
            tracker=self.create_extrinsic_tracker(),
            workspace_guide=self.workspace_guide,
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

        # Constraint factory reads the target config fresh from disk at calibration time
        targets_repo = self.targets_repository
        extrinsic_target_type = targets_repo.extrinsic_target_type
        if extrinsic_target_type == "aruco":

            def _build_constraints() -> ConstraintSet:
                return ConstraintSet.from_marker_set(targets_repo.load_aruco_marker_set())
        else:

            def _build_constraints() -> ConstraintSet:
                return ConstraintSet.from_charuco(targets_repo.load_extrinsic_charuco())

        constraint_factory = _build_constraints

        presenter = ExtrinsicCalibrationPresenter(
            task_manager=self.task_manager,
            camera_array=self.camera_array,
            image_points_path=image_points_path,
            existing_capture_volume=existing_capture_volume,
            constraint_factory=constraint_factory,
            project_settings=self.settings_repository,
            extrinsic_target_type=extrinsic_target_type,
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

        camera = self.camera_array.cameras[cam_id]
        if camera.rotation_count == rotation_count:
            return

        camera.rotation_count = rotation_count
        self.camera_repository.save(self.camera_array)
        logger.debug(f"Persisted camera rotation: cam_id {cam_id} -> {rotation_count * 90}°")
        self.calibration_changed.emit()

    def persist_intrinsic_calibration(
        self,
        output: IntrinsicCalibrationOutput,
        collected_points: list[tuple[int, PointPacket]] | None = None,
    ) -> None:
        """Persist intrinsic calibration result to ground truth.

        Updates the in-memory camera array and saves to disk. Also caches
        the calibration report for overlay restoration when switching cameras
        and persists it to disk for reload on app restart.

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
        self.calibration_changed.emit()
        self.status_changed.emit()

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

        Emits calibration_changed immediately (for UI refresh using in-memory state).
        Emits status_changed after save completes (for filesystem-based status checks).

        Also updates the main camera_array so that on restart, the calibrated
        extrinsics are available (enables tab and correct state detection).

        Args:
            capture_volume: The new CaptureVolume to store
        """
        self._capture_volume = capture_volume
        self.camera_array = capture_volume.camera_array  # Keep main camera_array in sync
        self.calibration_changed.emit()

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

    def cleanup(self) -> None:
        """Shutdown all background operations.

        Should be called when the application is closing to ensure threads
        are properly terminated. The TaskManager handles its own thread
        pool shutdown with configurable timeout.
        """
        logger.info("WorkspaceCoordinator cleanup initiated")
        self._recording_poll_timer.stop()
        self.task_manager.shutdown(timeout_ms=5000)
        logger.info("WorkspaceCoordinator cleanup complete")
