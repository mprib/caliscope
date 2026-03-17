"""Presenter for extrinsic calibration workflow.

Manages the workflow from ImagePoints through bundle adjustment optimization
to calibrated camera poses. Wraps CaptureVolume operations with Qt signals
for UI integration.

State is computed from internal reality, never stored separately.
This prevents state/reality divergence.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.build_paired_pose_network import (
    build_paired_pose_network,
)
from caliscope.core.coverage_analysis import compute_coverage_matrix
from caliscope.core.point_data import ImagePoints
from caliscope.core.capture_volume import CaptureVolume
from caliscope.repositories.project_settings_repository import ProjectSettingsRepository
from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_state import TaskState

logger = logging.getLogger(__name__)


class ExtrinsicCalibrationState(Enum):
    """Workflow states for extrinsic calibration.

    States are computed from internal reality, not stored separately.
    This prevents state/reality divergence.
    """

    NEEDS_CALIBRATION = auto()  # Have ImagePoints path, need to calibrate
    CALIBRATING = auto()  # Background calibration/optimization running
    CALIBRATED = auto()  # Have capture volume, can refine


@dataclass(frozen=True)
class FilterPreviewData:
    """Data for filter UI showing translation between modes.

    Provides bidirectional preview:
    - threshold_at_percentile: percentile-to-remove -> pixel threshold
    - errors: raw error array for computing percentile at any threshold
    """

    total_observations: int
    mean_error: float
    # Maps percentile-to-remove -> pixel threshold
    threshold_at_percentile: dict[int, float]
    # Raw errors for computing reverse lookup (threshold -> percentile)
    errors: tuple[float, ...]

    @classmethod
    def empty(cls) -> FilterPreviewData:
        """Create empty preview data."""
        return cls(
            total_observations=0,
            mean_error=0.0,
            threshold_at_percentile={},
            errors=(),
        )

    def percent_above_threshold(self, threshold: float) -> float:
        """Compute what percentage of observations exceed the threshold."""
        if len(self.errors) == 0:
            return 0.0
        count_above = sum(1 for e in self.errors if e > threshold)
        return 100.0 * count_above / len(self.errors)


@dataclass(frozen=True)
class QualityPanelData:
    """Quality metrics for display in the UI.

    Contains reprojection error statistics and optimization metadata.
    """

    # Reprojection metrics
    overall_rmse: float
    n_observations: int
    n_world_points: int

    # Per-camera table rows: (cam_id, n_obs, rmse)
    camera_rows: list[tuple[int, int, float]]

    # Optimization metadata
    converged: bool
    iterations: int

    # Filter preview for UI
    filter_preview: FilterPreviewData | None


class ExtrinsicCalibrationPresenter(QObject):
    """Presenter for extrinsic calibration workflow.

    Manages the extraction of camera extrinsics from 2D observations.
    Coordinates bootstrap triangulation, bundle adjustment optimization,
    and coordinate frame transformations.

    Signals:
        state_changed: Emitted when computed state changes. View updates UI.
        progress_updated: Emitted during optimization with (percent, message).
        quality_updated: Emitted when metrics refresh after calibration.
        capture_volume_changed: Emitted when capture volume is updated (optimization, rotate, align).
            Contains the new CaptureVolume.
        view_model_updated: Emitted when 3D view needs refresh.
            Contains PlaybackViewModel.

    Usage:
        presenter = ExtrinsicCalibrationPresenter(
            task_manager, camera_array, image_points_path
        )
        presenter.run_calibration()  # Bootstrap + optimize
        # On completion: capture_volume_changed emitted with capture volume
    """

    # State signals
    state_changed = Signal(object)  # ExtrinsicCalibrationState

    # Progress signals
    progress_updated = Signal(int, str)  # (percent, message)

    # Result signals
    quality_updated = Signal(object)  # QualityPanelData
    volumetric_accuracy_updated = Signal(object)  # VolumetricScaleReport
    coverage_updated = Signal(object, object)  # (coverage_matrix, cam_id_labels)
    capture_volume_changed = Signal(object)  # CaptureVolume
    view_model_updated = Signal(object)  # PlaybackViewModel

    def __init__(
        self,
        task_manager: TaskManager,
        camera_array: CameraArray,
        image_points_path: Path,
        existing_capture_volume: CaptureVolume | None = None,
        project_settings: ProjectSettingsRepository | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            task_manager: TaskManager for background processing
            camera_array: Initial camera configuration (extrinsics may be unset)
            image_points_path: Path to image_points.csv from Phase 3
            existing_capture_volume: Pre-loaded CaptureVolume for restoring calibrated state.
                If provided, presenter starts in CALIBRATED state with visualization ready.
            project_settings: Repository for persisting 3D view appearance settings.
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._task_manager = task_manager
        self._camera_array = camera_array
        self._image_points_path = image_points_path
        self._project_settings = project_settings

        # Processing state (managed internally)
        self._capture_volume: CaptureVolume | None = existing_capture_volume
        self._task_handle: TaskHandle | None = None

        # Pre-loaded image points for initial coverage display
        self._initial_image_points: ImagePoints | None = None

        # View state
        self._current_sync_index: int = 0

        # Load image points for coverage display (from capture volume if available, else CSV)
        if existing_capture_volume is not None:
            self._initial_image_points = existing_capture_volume.image_points
            # Set initial sync index from capture volume
            sync_indices = existing_capture_volume.unique_sync_indices
            if len(sync_indices) > 0:
                self._current_sync_index = int(sync_indices[0])
        else:
            self._load_initial_image_points()

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    def _is_task_active(self) -> bool:
        """True if a task is submitted and not yet finished.

        Includes both PENDING (just submitted, worker thread not yet running)
        and RUNNING states. This prevents the race condition where checking
        only RUNNING misses the window between submit() and worker start.
        """
        return self._task_handle is not None and self._task_handle.state in (
            TaskState.PENDING,
            TaskState.RUNNING,
        )

    @property
    def state(self) -> ExtrinsicCalibrationState:
        """Compute current state from internal reality - never stale."""
        if self._is_task_active():
            return ExtrinsicCalibrationState.CALIBRATING

        if self._capture_volume is not None:
            return ExtrinsicCalibrationState.CALIBRATED

        return ExtrinsicCalibrationState.NEEDS_CALIBRATION

    @property
    def capture_volume(self) -> CaptureVolume | None:
        """Current capture volume (None before calibration)."""
        return self._capture_volume

    @property
    def current_sync_index(self) -> int:
        """Current frame index for 3D visualization."""
        return self._current_sync_index

    # -------------------------------------------------------------------------
    # Calibration Workflow (Implemented in 4.2)
    # -------------------------------------------------------------------------

    def run_calibration(self) -> None:
        """Bootstrap poses and run bundle adjustment.

        Loads image_points.csv, performs stereo bootstrap triangulation,
        then runs bundle adjustment optimization. Emits capture_volume_changed
        with the optimized capture volume.

        Can be called from NEEDS_CALIBRATION or CALIBRATED state.
        In CALIBRATED state, discards current capture volume first.
        """
        if self.state == ExtrinsicCalibrationState.CALIBRATING:
            logger.warning("Cannot run calibration: already running")
            return

        # Clear existing capture volume to allow re-calibration from CALIBRATED state
        self._capture_volume = None

        # Capture for closure - deepcopy camera_array since bootstrap mutates it
        image_points_path = self._image_points_path
        camera_array = deepcopy(self._camera_array)

        def worker(token: CancellationToken, handle: TaskHandle) -> CaptureVolume:
            return self._execute_calibration(image_points_path, camera_array, token, handle)

        self._task_handle = self._task_manager.submit(
            worker,
            name="Extrinsic calibration",
            auto_start=False,
        )
        # Use QueuedConnection because TaskHandle signals are emitted from worker threads.
        # Without explicit QueuedConnection, Qt uses DirectConnection (sender/receiver both
        # have main thread affinity), causing slots to run in the worker thread.
        self._task_handle.completed.connect(
            self._on_capture_volume_optimized,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.failed.connect(
            self._on_calibration_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.cancelled.connect(
            self._on_calibration_cancelled,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.progress_updated.connect(
            self.progress_updated,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_manager.start_task(self._task_handle.task_id)

        self._emit_state_changed()

    def _execute_calibration(
        self,
        image_points_path: Path,
        camera_array: CameraArray,
        token: CancellationToken,
        handle: TaskHandle,
    ) -> CaptureVolume:
        """Execute full calibration pipeline. Runs in background thread.

        Stages:
        1. Load ImagePoints from CSV
        2. Bootstrap camera poses via stereo pair network
        3. Triangulate initial 3D world points
        4. Run bundle adjustment optimization
        5. Filter worst 2.5% of observations
        6. Final optimization pass
        """
        handle.report_progress(5, "Loading image points")
        image_points = ImagePoints.from_csv(image_points_path)

        if token.is_cancelled:
            raise InterruptedError("Calibration cancelled")

        handle.report_progress(15, "Bootstrapping camera poses")
        pose_network = build_paired_pose_network(image_points, camera_array, method="pnp")
        pose_network.apply_to(camera_array)

        if token.is_cancelled:
            raise InterruptedError("Calibration cancelled")

        handle.report_progress(30, "Triangulating 3D points")
        world_points = image_points.triangulate(camera_array)

        handle.report_progress(40, "Building capture volume")
        capture_volume = CaptureVolume(camera_array, image_points, world_points)

        if token.is_cancelled:
            raise InterruptedError("Calibration cancelled")

        handle.report_progress(50, "Running initial optimization")
        optimized = capture_volume.optimize(ftol=1e-8, verbose=0)
        logger.info(f"Initial optimization RMSE: {optimized.reprojection_report.overall_rmse:.3f}px")

        if token.is_cancelled:
            raise InterruptedError("Calibration cancelled")

        handle.report_progress(75, "Filtering outliers (2.5%)")
        filtered = optimized.filter_by_percentile_error(2.5)

        if token.is_cancelled:
            raise InterruptedError("Calibration cancelled")

        handle.report_progress(85, "Final optimization")
        final = filtered.optimize(ftol=1e-8, verbose=0)
        logger.info(f"Final optimization RMSE: {final.reprojection_report.overall_rmse:.3f}px")

        handle.report_progress(100, "Complete")
        return final

    # -------------------------------------------------------------------------
    # Filtering Operations (Implemented in 4.3)
    # -------------------------------------------------------------------------

    def filter_by_percentile(self, percentile: float) -> None:
        """Filter worst N% of observations and re-optimize.

        Args:
            percentile: Percentage of worst observations to remove (0-100)
        """
        if self._capture_volume is None:
            return

        filtered = self._capture_volume.filter_by_percentile_error(percentile)
        logger.info(f"Filtered {percentile}% worst observations, {len(filtered.image_points.df)} remaining")
        self._submit_optimization(filtered)

    def filter_by_threshold(self, max_error_pixels: float) -> None:
        """Filter observations above threshold and re-optimize.

        Args:
            max_error_pixels: Maximum reprojection error in pixels to keep
        """
        if self._capture_volume is None:
            return

        filtered = self._capture_volume.filter_by_absolute_error(max_error_pixels)
        logger.info(f"Filtered to error < {max_error_pixels}px, {len(filtered.image_points.df)} remaining")
        self._submit_optimization(filtered)

    def get_filter_preview(self) -> FilterPreviewData:
        """Get error stats for filter UI.

        Returns data allowing the View to show translation between filter modes:
        - Percentile mode: "Removing 5% would remove observations > 1.23px"
        - Absolute mode: "Removing observations > 1.0px would filter 3.2%"
        """
        if self._capture_volume is None:
            return FilterPreviewData.empty()

        report = self._capture_volume.reprojection_report
        # Use to_numpy() for type safety - .values can return ExtensionArray
        errors = report.raw_errors["euclidean_error"].to_numpy()

        return FilterPreviewData(
            total_observations=len(errors),
            mean_error=float(np.mean(errors)),
            threshold_at_percentile={
                1: float(np.percentile(errors, 99)),
                2: float(np.percentile(errors, 98)),
                3: float(np.percentile(errors, 97)),
                5: float(np.percentile(errors, 95)),
                10: float(np.percentile(errors, 90)),
                15: float(np.percentile(errors, 85)),
                20: float(np.percentile(errors, 80)),
            },
            errors=tuple(float(e) for e in errors),
        )

    # -------------------------------------------------------------------------
    # Coordinate Frame Operations (Implemented in 4.3)
    # -------------------------------------------------------------------------

    def rotate(self, axis: str, degrees: float) -> None:
        """Rotate coordinate frame around axis.

        Args:
            axis: "x", "y", or "z"
            degrees: Rotation angle in degrees (positive = counter-clockwise)
        """
        if self._capture_volume is None:
            return

        # CaptureVolume.rotate() expects Literal["x", "y", "z"]
        # The domain method validates the axis value
        axis_typed: Literal["x", "y", "z"] = axis  # type: ignore[assignment]
        rotated = self._capture_volume.rotate(axis_typed, degrees)
        logger.info(f"Rotated coordinate frame {degrees}° around {axis}-axis")
        self._update_capture_volume(rotated)

    def align_to_origin(self, sync_index: int) -> None:
        """Set world origin to board position at sync_index.

        Args:
            sync_index: Frame index where board position defines origin
        """
        if self._capture_volume is None:
            return

        aligned = self._capture_volume.align_to_object(sync_index)
        logger.info(f"Aligned world origin to object at sync_index={sync_index}")

        self._update_capture_volume(aligned)

    # -------------------------------------------------------------------------
    # View Control (Implemented in 4.3)
    # -------------------------------------------------------------------------

    def set_sync_index(self, index: int) -> None:
        """Update current frame for 3D view.

        Note: This only updates the presenter's internal tracking of the current
        sync index. It does NOT emit view_model_updated - the view should directly
        update the PyVista widget for efficient frame-by-frame rendering.

        Args:
            index: Sync index to display
        """
        if self._capture_volume is None:
            return

        # Clamp to valid range
        sync_indices = self._capture_volume.unique_sync_indices
        if len(sync_indices) == 0:
            return

        if index < sync_indices.min():
            index = int(sync_indices.min())
        elif index > sync_indices.max():
            index = int(sync_indices.max())

        self._current_sync_index = index
        # Note: No view_model_updated emission here - frame changes are
        # handled directly by the view calling widget.set_sync_index()

    # -------------------------------------------------------------------------
    # Scene Appearance Settings
    # -------------------------------------------------------------------------

    def save_camera_size_multiplier(self, multiplier: float) -> None:
        """Persist the camera frustum size multiplier to project settings."""
        if self._project_settings is not None:
            self._project_settings.set_scene_camera_size_multiplier(multiplier)

    def save_grid_size_multiplier(self, multiplier: float) -> None:
        """Persist the floor grid size multiplier to project settings."""
        if self._project_settings is not None:
            self._project_settings.set_scene_grid_size_multiplier(multiplier)

    def save_sphere_size_multiplier(self, multiplier: float) -> None:
        """Persist the point sphere size multiplier to project settings."""
        if self._project_settings is not None:
            self._project_settings.set_scene_sphere_size_multiplier(multiplier)

    def get_camera_size_multiplier(self) -> float:
        """Load camera frustum size multiplier from project settings (default: 1.0)."""
        if self._project_settings is not None:
            return self._project_settings.get_scene_camera_size_multiplier()
        return 1.0

    def get_grid_size_multiplier(self) -> float:
        """Load floor grid size multiplier from project settings (default: 1.0)."""
        if self._project_settings is not None:
            return self._project_settings.get_scene_grid_size_multiplier()
        return 1.0

    def get_sphere_size_multiplier(self) -> float:
        """Load point sphere size multiplier from project settings (default: 1.0)."""
        if self._project_settings is not None:
            return self._project_settings.get_scene_sphere_size_multiplier()
        return 1.0

    # -------------------------------------------------------------------------
    # Cancellation
    # -------------------------------------------------------------------------

    def cancel_calibration(self) -> None:
        """Request cancellation of running calibration.

        Unlike cleanup(), this leaves the task handle intact so the
        cancelled callback can drive the state transition naturally.
        The _on_calibration_cancelled slot will clear the handle and
        emit state_changed.
        """
        if self._task_handle is not None and self._is_task_active():
            logger.info("Cancel requested by user")
            self._task_handle.cancel()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up resources. Call before discarding presenter."""
        if self._task_handle is not None:
            self._task_handle.cancel()
            self._task_handle = None

    # -------------------------------------------------------------------------
    # Private: Task Callbacks
    # -------------------------------------------------------------------------

    def _on_capture_volume_optimized(self, capture_volume: CaptureVolume) -> None:
        """Handle successful calibration/optimization completion."""
        logger.info(f"Calibration complete. RMSE: {capture_volume.reprojection_report.overall_rmse:.3f}px")

        self._capture_volume = capture_volume
        self._task_handle = None

        # Set initial sync index to first available frame
        sync_indices = capture_volume.unique_sync_indices
        if len(sync_indices) > 0:
            self._current_sync_index = int(sync_indices[0])

        self._emit_state_changed()
        self._refresh_quality_panel()
        self._refresh_coverage()
        self._refresh_view_model()
        self._refresh_volumetric_accuracy()
        self.capture_volume_changed.emit(capture_volume)

    def _on_calibration_failed(self, exc_type: str, message: str) -> None:
        """Handle calibration failure."""
        logger.error(f"Calibration failed: {exc_type}: {message}")
        self._task_handle = None
        self._emit_state_changed()

    def _on_calibration_cancelled(self) -> None:
        """Handle calibration cancellation."""
        logger.info("Calibration cancelled")
        self._task_handle = None
        self._emit_state_changed()

    # -------------------------------------------------------------------------
    # Private: State Management
    # -------------------------------------------------------------------------

    def _emit_state_changed(self) -> None:
        """Emit state_changed signal with current computed state."""
        current_state = self.state
        logger.debug(f"State changed to {current_state}")
        self.state_changed.emit(current_state)

    def _refresh_quality_panel(self) -> None:
        """Build and emit quality panel data from current capture volume."""
        if self._capture_volume is None:
            return

        report = self._capture_volume.reprojection_report
        status = self._capture_volume.optimization_status

        # Build per-camera rows: (cam_id, n_obs, rmse)
        camera_rows: list[tuple[int, int, float]] = []
        for cam_id in sorted(report.by_camera.keys()):
            n_obs = int((self._capture_volume.image_points.df["cam_id"] == cam_id).sum())
            rmse = report.by_camera[cam_id]
            camera_rows.append((cam_id, n_obs, rmse))

        quality_data = QualityPanelData(
            overall_rmse=report.overall_rmse,
            n_observations=report.n_observations_matched,
            n_world_points=report.n_points,
            camera_rows=camera_rows,
            converged=status.converged if status else False,
            iterations=status.iterations if status else 0,
            filter_preview=None,  # Populated on demand via get_filter_preview()
        )

        self.quality_updated.emit(quality_data)

    def _refresh_view_model(self) -> None:
        """Build and emit PlaybackViewModel for 3D visualization."""
        if self._capture_volume is None:
            return

        # Import here to avoid circular dependency at module level
        from caliscope.gui.view_models.playback_view_model import PlaybackViewModel

        view_model = PlaybackViewModel(
            camera_array=self._capture_volume.camera_array,
            world_points=self._capture_volume.world_points,
        )
        self.view_model_updated.emit(view_model)

    def _refresh_volumetric_accuracy(self) -> None:
        """Compute and emit volumetric scale accuracy across all valid frames.

        Scans all frames with >=4 corners and obj_loc data, computing distance
        RMSE at each frame. Returns empty report if no valid frames exist
        (normal pre-alignment state).
        """
        if self._capture_volume is None:
            return

        report = self._capture_volume.compute_volumetric_scale_accuracy()
        if report.n_frames_sampled > 0:
            logger.info(
                f"Volumetric scale accuracy: pooled RMSE={report.pooled_rmse_mm:.2f}mm, "
                f"{report.n_frames_sampled} frames sampled"
            )
        self.volumetric_accuracy_updated.emit(report)

    def _load_initial_image_points(self) -> None:
        """Load ImagePoints for initial coverage display.

        Called during __init__ to show coverage before calibration runs.
        The ImagePoints are stored to avoid reloading during calibration.
        """
        try:
            self._initial_image_points = ImagePoints.from_csv(self._image_points_path)
            logger.debug(f"Loaded {len(self._initial_image_points.df)} initial image points")
        except Exception as e:
            logger.warning(f"Could not load initial image points: {e}")
            self._initial_image_points = None

    def emit_initial_state(self) -> None:
        """Emit initial state for UI display after signal connections.

        Call this after connecting signals. Emits:
        - Coverage matrix (always, from ImagePoints)
        - Quality data and view model (if existing capture volume loaded)

        This enables the view to show the correct initial state:
        - Fresh start: coverage heatmap, "Calibrate" button
        - Restored session: 3D visualization, quality metrics, "Calibrate" button
        """
        # Always emit coverage from initial image points
        self._refresh_initial_coverage()

        # If we have an existing capture volume, emit quality and view model for 3D viz
        if self._capture_volume is not None:
            logger.info("Emitting initial state from existing capture volume")
            self._refresh_quality_panel()
            self._refresh_coverage()  # Use capture volume's coverage (may differ after filtering)
            self._refresh_view_model()
            self._refresh_volumetric_accuracy()

    def emit_initial_coverage(self) -> None:
        """Deprecated: Use emit_initial_state() instead."""
        self.emit_initial_state()

    def _refresh_initial_coverage(self) -> None:
        """Emit coverage matrix from pre-loaded ImagePoints.

        Internal method - use emit_initial_state() from the view.
        Uses cam_ids discovered from ImagePoints data (not posed cameras).
        """
        if self._initial_image_points is None:
            return

        df = self._initial_image_points.df
        if len(df) == 0:
            return

        # Build cam_id-to-index mapping from actual data
        actual_cam_ids = sorted(df["cam_id"].unique())
        cam_id_to_index = {int(cam_id): idx for idx, cam_id in enumerate(actual_cam_ids)}

        coverage = compute_coverage_matrix(self._initial_image_points, cam_id_to_index)
        labels = [f"C{c}" for c in actual_cam_ids]

        self.coverage_updated.emit(coverage, labels)

    def _refresh_coverage(self) -> None:
        """Emit coverage matrix data for heatmap visualization.

        Computes pairwise observation counts between all posed cameras.
        Labels use camera IDs (C1, C2, etc.) matching the camera array.
        """
        if self._capture_volume is None:
            return

        camera_array = self._capture_volume.camera_array
        cam_id_to_index = camera_array.posed_cam_id_to_index

        if not cam_id_to_index:
            logger.debug("No posed cameras for coverage matrix")
            return

        coverage = compute_coverage_matrix(self._capture_volume.image_points, cam_id_to_index)
        labels = [f"C{c}" for c in sorted(cam_id_to_index.keys())]

        self.coverage_updated.emit(coverage, labels)

    def _submit_optimization(self, capture_volume: CaptureVolume) -> None:
        """Submit capture volume optimization as background task.

        Used by filter methods to avoid duplicating task setup code.
        After filtering, the capture volume needs re-optimization to update
        camera extrinsics and world points.

        Args:
            capture_volume: Filtered CaptureVolume to optimize
        """
        if self._is_task_active():
            logger.warning("Cannot start optimization: task already running")
            return

        def worker(token: CancellationToken, handle: TaskHandle) -> CaptureVolume:
            handle.report_progress(10, "Running optimization")
            optimized = capture_volume.optimize(ftol=1e-8, verbose=0)
            handle.report_progress(100, "Complete")
            logger.info(f"Post-filter optimization RMSE: {optimized.reprojection_report.overall_rmse:.3f}px")
            return optimized

        self._task_handle = self._task_manager.submit(worker, name="Optimize capture volume", auto_start=False)
        # Use QueuedConnection - TaskHandle signals emitted from worker threads
        self._task_handle.completed.connect(
            self._on_capture_volume_optimized,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.failed.connect(
            self._on_calibration_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.cancelled.connect(
            self._on_calibration_cancelled,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.progress_updated.connect(
            self.progress_updated,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_manager.start_task(self._task_handle.task_id)
        self._emit_state_changed()

    def _update_capture_volume(self, capture_volume: CaptureVolume) -> None:
        """Update internal capture volume and emit signals.

        Called after synchronous capture-volume-modifying operations (rotate, align).
        These operations don't change the optimization status, just transform
        the coordinate frame.

        Args:
            capture_volume: New CaptureVolume after transformation
        """
        self._capture_volume = capture_volume
        self._emit_state_changed()
        self._refresh_quality_panel()
        self._refresh_view_model()
        self._refresh_volumetric_accuracy()
        self.capture_volume_changed.emit(capture_volume)
