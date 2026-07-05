"""Presenter for extrinsic calibration workflow.

Manages the workflow from ImagePoints through bundle adjustment optimization
to calibrated camera poses. Wraps CaptureVolume operations with Qt signals
for UI integration.

State is computed from internal reality, never stored separately.
This prevents state/reality divergence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.calibrate_extrinsics import (
    ExtrinsicCalibrationResult,
    calibrate_extrinsics,
    refresh_result,
)
from caliscope.core.coverage_analysis import compute_coverage_matrix
from caliscope.core.point_data import ImagePoints
from caliscope.core.capture_volume import CaptureVolume, OptimizationStatus
from caliscope.core.constraints import ConstraintSet
from caliscope.core.scale_accuracy import VolumetricScaleReport, compute_depth_ratios
from caliscope.core.workflow_status import StepStatus
from caliscope.repositories.calibration_targets_repository import ExtrinsicTargetType
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
class CameraQualityRow:
    cam_id: int
    n_observations: int
    rmse_px: float
    f_px: float
    k1: float
    k2: float
    source: str  # "refined" | "locked" | "estimated" | "—"
    depth_ratio: float | None
    delta_f: float | None  # final - initial, None when source is estimated/unknown
    delta_k1: float | None
    delta_k2: float | None


@dataclass(frozen=True)
class MarkerQualityRow:
    object_id: int
    is_static: bool
    reprojection_rmse_px: float
    relative_rmse_pct: float
    rmse_mm: float
    n_pairs: int


@dataclass(frozen=True)
class CalibrationQualityData:
    """Calibration quality metrics for display in the UI.

    Combines reprojection error, scale accuracy, and per-camera/per-marker
    breakdowns into a single display model consumed by the quality report.
    """

    overall_rmse_px: float
    n_observations: int
    n_world_points: int
    converged: bool | None  # None = status unknown (session restore)
    iterations: int
    distance_error_pct: float | None  # VolumetricScaleReport.pooled_relative_rmse_pct, None when no pairs
    distance_error_mm: float | None  # VolumetricScaleReport.pooled_rmse_mm
    moving_error_pct: float | None
    static_error_pct: float | None
    warnings: list[str]
    filter_summary: str | None
    camera_rows: list[CameraQualityRow]
    marker_rows: list[MarkerQualityRow]  # empty -> hide Markers tab
    cameras: dict[int, CameraData]  # for LensModelDialog
    extrinsic_dir: Path  # for LensModelDialog


@dataclass(frozen=True)
class OriginOption:
    object_id: int
    label: str
    is_static: bool


@dataclass(frozen=True)
class CalibrationStepData:
    extract: tuple[StepStatus, str]  # status + detail text
    calibrate: tuple[StepStatus, str]
    origin: tuple[StepStatus, str]  # only AVAILABLE or COMPLETE ever emitted


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
    quality_updated = Signal(object)  # CalibrationQualityData
    volumetric_accuracy_updated = Signal(object)  # VolumetricScaleReport
    coverage_updated = Signal(object, object)  # (coverage_matrix, cam_id_labels)
    capture_volume_changed = Signal(object)  # CaptureVolume
    calibration_result_updated = Signal(object)  # ExtrinsicCalibrationResult
    workflow_updated = Signal(object)  # CalibrationStepData
    view_model_updated = Signal(object)  # PlaybackViewModel

    def __init__(
        self,
        task_manager: TaskManager,
        camera_array: CameraArray,
        image_points_path: Path,
        existing_capture_volume: CaptureVolume | None = None,
        constraint_factory: Callable[[], ConstraintSet | None] | None = None,
        project_settings: ProjectSettingsRepository | None = None,
        extrinsic_target_type: ExtrinsicTargetType | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            task_manager: TaskManager for background processing
            camera_array: Initial camera configuration (extrinsics may be unset)
            image_points_path: Path to image_points.csv from Phase 3
            existing_capture_volume: Pre-loaded CaptureVolume for restoring calibrated state.
                If provided, presenter starts in CALIBRATED state with visualization ready.
            constraint_factory: Called at calibration time to get current constraints from disk.
            project_settings: Repository for persisting 3D view appearance settings.
            extrinsic_target_type: Which target produced the calibration ("charuco" or
                "aruco"), used only to label the single-object origin option. None when
                unknown (e.g. tests constructing the presenter directly).
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._task_manager = task_manager
        self._camera_array = camera_array
        self._image_points_path = image_points_path
        self._constraint_factory = constraint_factory
        self._project_settings = project_settings
        self._extrinsic_target_type = extrinsic_target_type

        # Processing state (managed internally)
        self._capture_volume: CaptureVolume | None = existing_capture_volume
        self._calibration_result: ExtrinsicCalibrationResult | None = None
        self._refine_intrinsics: bool = True
        self._task_handle: TaskHandle | None = None
        self._filter_summary: str | None = None
        self._latest_scale_report: VolumetricScaleReport | None = None

        # Pre-loaded image points for initial coverage display
        self._initial_image_points: ImagePoints | None = None

        # View state
        self._current_sync_index: int = 0

        # Load image points for coverage display (from capture volume if available, else CSV)
        if existing_capture_volume is not None:
            self._initial_image_points = existing_capture_volume.image_points
            # Set initial sync index, skipping STATIC_SYNC_INDEX (-1)
            from caliscope.core.point_data import STATIC_SYNC_INDEX

            sync_indices = existing_capture_volume.unique_sync_indices
            valid = sync_indices[sync_indices != STATIC_SYNC_INDEX]
            if len(valid) > 0:
                self._current_sync_index = int(valid[0])
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

    @property
    def intrinsics_available(self) -> bool:
        """Whether all active cameras have intrinsic calibration."""
        return all(cam.matrix is not None for cam in self._camera_array.cameras.values())

    @property
    def refine_intrinsics(self) -> bool:
        """Current refine-intrinsics setting. Forced True when any camera lacks intrinsics."""
        if not self.intrinsics_available:
            return True
        if self._project_settings is not None:
            return self._project_settings.get_refine_intrinsics()
        return True

    def set_refine_intrinsics(self, enabled: bool) -> None:
        """Persist the refine-intrinsics toggle."""
        if self._project_settings is not None:
            self._project_settings.set_refine_intrinsics(enabled)

    @property
    def current_origin_object_id(self) -> int | None:
        """Currently persisted origin marker id, or None if no origin has been set."""
        if self._project_settings is None:
            return None
        return self._project_settings.get_origin_object_id()

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

        # Clear existing state to allow re-calibration from CALIBRATED state
        self._capture_volume = None
        self._calibration_result = None
        self._filter_summary = None

        # Clear persisted origin — the old transform is invalidated by recalibration
        if self._project_settings is not None:
            self._project_settings.set_origin_object_id(None)
            self._project_settings.set_origin_sync_index(None)

        # Capture for closure - deepcopy camera_array since bootstrap mutates it
        image_points_path = self._image_points_path
        camera_array = deepcopy(self._camera_array)

        self._refine_intrinsics = self.refine_intrinsics

        def worker(token: CancellationToken, handle: TaskHandle) -> ExtrinsicCalibrationResult:
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
            self._on_calibration_completed,
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
        self._refresh_workflow_strip()

    def _execute_calibration(
        self,
        image_points_path: Path,
        camera_array: CameraArray,
        token: CancellationToken,
        handle: TaskHandle,
    ) -> ExtrinsicCalibrationResult:
        """Execute full calibration pipeline. Runs in background thread."""
        image_points = ImagePoints.from_csv(image_points_path)
        constraints = self._constraint_factory() if self._constraint_factory else None

        return calibrate_extrinsics(
            image_points,
            camera_array,
            constraints,
            refine_intrinsics=self._refine_intrinsics,
            cancellation_token=token,
            progress=lambda pct, msg: handle.report_progress(pct, msg),
        )

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

        original_count = len(self._capture_volume.image_points.df)
        filtered = self._capture_volume.filter_by_percentile_error(percentile)
        remaining_count = len(filtered.image_points.df)
        remaining_pct = 100.0 * remaining_count / original_count if original_count > 0 else 0.0
        keep_percentile = 100 - percentile
        self._filter_summary = f"{remaining_pct:.0f}% remaining after {keep_percentile:.0f}th percentile filter"
        logger.info(f"Filtered {percentile}% worst observations, {remaining_count} remaining")
        self._submit_optimization(filtered)

    def filter_by_threshold(self, max_error_pixels: float) -> None:
        """Filter observations above threshold and re-optimize.

        Args:
            max_error_pixels: Maximum reprojection error in pixels to keep
        """
        if self._capture_volume is None:
            return

        original_count = len(self._capture_volume.image_points.df)
        filtered = self._capture_volume.filter_by_absolute_error(max_error_pixels)
        remaining_count = len(filtered.image_points.df)
        remaining_pct = 100.0 * remaining_count / original_count if original_count > 0 else 0.0
        self._filter_summary = f"{remaining_pct:.0f}% remaining after {max_error_pixels:.2f}px threshold filter"
        logger.info(f"Filtered to error < {max_error_pixels}px, {remaining_count} remaining")
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

    def align_to_origin(self, object_id: int, sync_index: int | None) -> None:
        """Set world origin to a marker's position.

        Args:
            object_id: Marker/object whose local frame defines the new origin
            sync_index: Frame index where the marker's position defines the origin,
                or None to use a static marker's fixed pose
        """
        if self._capture_volume is None:
            return

        aligned = self._capture_volume.align_to_object(sync_index=sync_index, object_id=object_id)
        logger.info(
            f"Aligned world origin to object_id={object_id} "
            f"at sync_index={sync_index if sync_index is not None else 'static'}"
        )

        if self._project_settings is not None:
            self._project_settings.set_origin_object_id(object_id)
            self._project_settings.set_origin_sync_index(sync_index)

        self._update_capture_volume(aligned)

    def get_origin_options(self) -> list[OriginOption]:
        """Return available origin markers from the calibrated volume, static markers first."""
        if self._capture_volume is None:
            return []

        wp_df = self._capture_volume.world_points.df
        object_ids = sorted(int(oid) for oid in wp_df["object_id"].unique())

        constraints = self._capture_volume.constraints
        static_ids = constraints.static_object_ids if constraints else frozenset()

        options: list[OriginOption] = []
        for oid in object_ids:
            if oid in static_ids:
                options.append(OriginOption(object_id=oid, label=f"marker {oid} (static)", is_static=True))
        for oid in object_ids:
            if oid not in static_ids:
                if self._is_board_origin(object_ids, static_ids):
                    options.append(OriginOption(object_id=oid, label="board", is_static=False))
                else:
                    options.append(OriginOption(object_id=oid, label=f"marker {oid}", is_static=False))

        return options

    def _is_board_origin(self, object_ids: list[int], static_ids: frozenset[int]) -> bool:
        """Whether the single non-static origin option should be labeled "board".

        Charuco calibration always produces exactly one object_id (0) with
        board-geometry constraints, so it is labeled "board" precisely when
        the target type is known to be charuco. When the target type is
        unknown (e.g. a presenter constructed directly in a test), fall back
        to the pre-existing heuristic: exactly one object and no static
        markers. That heuristic also matches a single-marker ArUco
        calibration — a low-stakes edge case (it picks up the "board" label
        instead of "marker 0") accepted because the target type is normally
        available from the coordinator.
        """
        if self._extrinsic_target_type is not None:
            return self._extrinsic_target_type == "charuco"
        return len(object_ids) == 1 and not static_ids

    def is_object_visible_at(self, object_id: int, sync_index: int) -> bool:
        """Check whether the given object has a triangulated world point at the given sync_index."""
        if self._capture_volume is None:
            return False
        wp_df = self._capture_volume.world_points.df
        mask = (wp_df["object_id"] == object_id) & (wp_df["sync_index"] == sync_index)
        return bool(mask.any())

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

    def _on_calibration_completed(self, result: ExtrinsicCalibrationResult) -> None:
        """Handle successful full calibration completion."""
        capture_volume = result.capture_volume
        logger.info(f"Calibration complete. RMSE: {capture_volume.reprojection_report.overall_rmse:.3f}px")

        self._calibration_result = result
        self._capture_volume = capture_volume
        self._task_handle = None

        sync_indices = capture_volume.unique_sync_indices
        if len(sync_indices) > 0:
            self._current_sync_index = int(sync_indices[0])

        self._emit_state_changed()
        self._refresh_quality_panel()
        self._refresh_coverage()
        self._refresh_view_model()
        self._refresh_volumetric_accuracy()
        self._refresh_workflow_strip()
        self.capture_volume_changed.emit(capture_volume)
        self.calibration_result_updated.emit(result)

    def _on_reoptimization_completed(self, capture_volume: CaptureVolume) -> None:
        """Handle successful filter re-optimization completion."""
        logger.info(f"Re-optimization complete. RMSE: {capture_volume.reprojection_report.overall_rmse:.3f}px")

        self._capture_volume = capture_volume
        self._task_handle = None

        if self._calibration_result is not None:
            self._calibration_result = refresh_result(self._calibration_result, capture_volume)

        self._emit_state_changed()
        self._refresh_quality_panel()
        self._refresh_coverage()
        self._refresh_view_model()
        self._refresh_volumetric_accuracy()
        self._refresh_workflow_strip()
        self.capture_volume_changed.emit(capture_volume)

        if self._calibration_result is not None:
            self.calibration_result_updated.emit(self._calibration_result)

    def _on_calibration_failed(self, exc_type: str, message: str) -> None:
        """Handle calibration failure."""
        logger.error(f"Calibration failed: {exc_type}: {message}")
        self._task_handle = None
        self._filter_summary = None
        self._emit_state_changed()
        self._refresh_workflow_strip()

    def _on_calibration_cancelled(self) -> None:
        """Handle calibration cancellation."""
        logger.info("Calibration cancelled")
        self._task_handle = None
        self._filter_summary = None
        self._emit_state_changed()
        self._refresh_workflow_strip()

    # -------------------------------------------------------------------------
    # Private: State Management
    # -------------------------------------------------------------------------

    def _emit_state_changed(self) -> None:
        """Emit state_changed signal with current computed state."""
        current_state = self.state
        logger.debug(f"State changed to {current_state}")
        self.state_changed.emit(current_state)

    def _build_camera_rows(
        self,
        cv: CaptureVolume,
        depth_ratios: dict[int, float],
    ) -> list[CameraQualityRow]:
        """Build per-camera quality rows, tagging intrinsic source when a result is available."""
        report = cv.reprojection_report
        result = self._calibration_result

        anchors: dict[int, tuple[float, float, float]] = {}
        synthesized_ids: frozenset[int] = frozenset()
        if result is not None:
            synthesized_ids = result.synthesized_cam_ids
            for est in result.intrinsic_estimates:
                anchors[est.cam_id] = (est.f_initial, est.k1_initial, est.k2_initial)

        rows: list[CameraQualityRow] = []
        for cam_id in sorted(report.by_camera.keys()):
            cam = cv.camera_array.cameras.get(cam_id)
            n_obs = int((cv.image_points.df["cam_id"] == cam_id).sum())
            rmse = report.by_camera[cam_id]
            f_px = float(cam.matrix[0, 0]) if cam is not None and cam.matrix is not None else 0.0
            k1 = float(cam.distortions[0]) if cam is not None and cam.distortions is not None else 0.0
            k2 = float(cam.distortions[1]) if cam is not None and cam.distortions is not None else 0.0

            delta_f: float | None = None
            delta_k1: float | None = None
            delta_k2: float | None = None
            if result is None:
                source = "—"
            elif cam_id in synthesized_ids:
                source = "estimated"
            elif cam_id in anchors:
                source = "refined" if self._refine_intrinsics else "locked"
                f_init, k1_init, k2_init = anchors[cam_id]
                delta_f = f_px - f_init
                delta_k1 = k1 - k1_init
                delta_k2 = k2 - k2_init
            else:
                source = "—"

            dr = depth_ratios.get(cam_id)
            depth_ratio = None if dr is None or np.isnan(dr) else dr

            rows.append(
                CameraQualityRow(
                    cam_id=cam_id,
                    n_observations=n_obs,
                    rmse_px=rmse,
                    f_px=f_px,
                    k1=k1,
                    k2=k2,
                    source=source,
                    depth_ratio=depth_ratio,
                    delta_f=delta_f,
                    delta_k1=delta_k1,
                    delta_k2=delta_k2,
                )
            )
        return rows

    def _build_marker_rows(self, scale_report: VolumetricScaleReport) -> list[MarkerQualityRow]:
        """Build per-marker quality rows from pooled scale accuracy and reprojection metrics."""
        if not scale_report.frame_errors:
            return []

        per_object_pct = scale_report.per_object_relative_rmse_pct
        sse_by_object: dict[int, float] = {}
        pairs_by_object: dict[int, int] = {}
        for fe in scale_report.frame_errors:
            sse_by_object[fe.object_id] = sse_by_object.get(fe.object_id, 0.0) + fe.sum_squared_errors_m2
            pairs_by_object[fe.object_id] = pairs_by_object.get(fe.object_id, 0) + fe.n_distance_pairs

        # Per-object reprojection RMSE from raw errors
        reproj_by_object: dict[int, float] = {}
        if self._capture_volume is not None:
            raw = self._capture_volume.reprojection_report.raw_errors
            for oid_key, group in raw.groupby("object_id"):
                oid_int = int(str(oid_key))
                errors = np.asarray(group["euclidean_error"], dtype=np.float64)
                reproj_by_object[oid_int] = float(np.sqrt(np.mean(errors**2)))

        rows: list[MarkerQualityRow] = []
        for object_id in sorted(per_object_pct.keys()):
            pairs = pairs_by_object.get(object_id, 0)
            rmse_mm = float(np.sqrt(sse_by_object[object_id] / pairs) * 1000) if pairs > 0 else 0.0
            rows.append(
                MarkerQualityRow(
                    object_id=object_id,
                    is_static=object_id in scale_report.static_object_ids,
                    reprojection_rmse_px=reproj_by_object.get(object_id, 0.0),
                    relative_rmse_pct=per_object_pct[object_id],
                    rmse_mm=rmse_mm,
                    n_pairs=pairs,
                )
            )
        return rows

    def _build_warnings(
        self, status: OptimizationStatus | None, result: ExtrinsicCalibrationResult | None
    ) -> list[str]:
        """Build warning strings from optimization status and calibration result."""
        warnings: list[str] = []
        if status is not None and not status.converged:
            warnings.append("Optimization did not converge")

        if result is not None:
            for bw in result.bound_warnings:
                warnings.append(f"Camera {bw.cam_id}: {bw.parameter} hit {bw.bound} bound at {bw.value:.1f}")
            if result.dropped_static_markers:
                ids_str = ", ".join(str(m) for m in result.dropped_static_markers)
                warnings.append(
                    f"Markers {ids_str} labeled static but showed too much positional "
                    f"variance — removed from static constraint set"
                )
        return warnings

    def _refresh_quality_panel(self) -> None:
        """Build and emit calibration quality data from current capture volume."""
        if self._capture_volume is None:
            return

        cv = self._capture_volume
        report = cv.reprojection_report
        status = cv.optimization_status
        result = self._calibration_result

        scale_report = cv.compute_volumetric_scale_accuracy()
        self._latest_scale_report = scale_report

        depth_ratios = compute_depth_ratios(cv)
        camera_rows = self._build_camera_rows(cv, depth_ratios)
        marker_rows = self._build_marker_rows(scale_report)
        warnings = self._build_warnings(status, result)

        moving_pct, static_pct = scale_report.split_relative_rmse_pct
        has_pairs = len(scale_report.frame_errors) > 0

        cameras = {cam.cam_id: cam for cam in cv.camera_array.cameras.values()}
        # Extrinsic dir: image_points_path is .../extrinsic/TRACKER/image_points.csv
        extrinsic_dir = self._image_points_path.parent.parent

        quality_data = CalibrationQualityData(
            overall_rmse_px=report.overall_rmse,
            n_observations=report.n_observations_matched,
            n_world_points=report.n_points,
            converged=status.converged if status else None,
            iterations=status.iterations if status else 0,
            distance_error_pct=scale_report.pooled_relative_rmse_pct if has_pairs else None,
            distance_error_mm=scale_report.pooled_rmse_mm if has_pairs else None,
            moving_error_pct=moving_pct,
            static_error_pct=static_pct,
            warnings=warnings,
            filter_summary=self._filter_summary,
            camera_rows=camera_rows,
            marker_rows=marker_rows,
            cameras=cameras,
            extrinsic_dir=extrinsic_dir,
        )

        self.quality_updated.emit(quality_data)

    def _refresh_workflow_strip(self) -> None:
        """Build and emit workflow step data for the extract/calibrate/origin strip."""
        # Extract step
        if self._initial_image_points is not None:
            n_obs = len(self._initial_image_points.df)
            extract = (StepStatus.COMPLETE, f"{n_obs:,} observations")
        else:
            extract = (StepStatus.NOT_STARTED, "run extraction on the Cameras tab")

        # Calibrate step
        state = self.state
        if state == ExtrinsicCalibrationState.CALIBRATED:
            assert self._capture_volume is not None
            rmse = self._capture_volume.reprojection_report.overall_rmse
            calibrate = (StepStatus.COMPLETE, f"{rmse:.2f} px RMSE")
        elif state == ExtrinsicCalibrationState.CALIBRATING:
            calibrate = (StepStatus.AVAILABLE, "calibrating…")
        else:
            calibrate = (StepStatus.AVAILABLE, "")

        # Origin step — only AVAILABLE or COMPLETE ever emitted
        origin_id = self._project_settings.get_origin_object_id() if self._project_settings else None
        if origin_id is not None:
            origin_sync = self._project_settings.get_origin_sync_index() if self._project_settings else None
            is_static = (
                self._capture_volume is not None
                and self._capture_volume.constraints is not None
                and origin_id in self._capture_volume.constraints.static_object_ids
            )
            if is_static or origin_sync is None:
                label = f"marker {origin_id} (static)"
            else:
                label = f"marker {origin_id} @ frame {origin_sync}"
            origin = (StepStatus.COMPLETE, label)
        else:
            origin = (StepStatus.AVAILABLE, "")

        self.workflow_updated.emit(CalibrationStepData(extract=extract, calibrate=calibrate, origin=origin))

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
        """Emit volumetric scale accuracy across all valid frames.

        Reuses the report computed by the preceding _refresh_quality_panel() call
        (self._latest_scale_report) to avoid recomputing it. Falls back to computing
        it directly if called without a preceding quality refresh.
        """
        if self._capture_volume is None:
            return

        report = self._latest_scale_report
        if report is None:
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

        self._refresh_workflow_strip()

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

        refine = self.refine_intrinsics

        def worker(token: CancellationToken, handle: TaskHandle) -> CaptureVolume:
            handle.report_progress(10, "Running optimization")
            optimized = capture_volume.optimize(ftol=1e-8, verbose=0, refine_intrinsics=refine)
            handle.report_progress(100, "Complete")
            logger.info(f"Post-filter optimization RMSE: {optimized.reprojection_report.overall_rmse:.3f}px")
            return optimized

        self._task_handle = self._task_manager.submit(worker, name="Optimize capture volume", auto_start=False)
        # Use QueuedConnection - TaskHandle signals emitted from worker threads
        self._task_handle.completed.connect(
            self._on_reoptimization_completed,
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
        self._refresh_workflow_strip()

    def _update_capture_volume(self, capture_volume: CaptureVolume) -> None:
        """Update internal capture volume and emit signals.

        Called after synchronous capture-volume-modifying operations (rotate, align).
        These operations don't change the optimization status, just transform
        the coordinate frame.

        Args:
            capture_volume: New CaptureVolume after transformation
        """
        self._capture_volume = capture_volume

        if self._calibration_result is not None:
            self._calibration_result = refresh_result(self._calibration_result, capture_volume)

        self._emit_state_changed()
        self._refresh_quality_panel()
        self._refresh_view_model()
        self._refresh_volumetric_accuracy()
        self._refresh_workflow_strip()
        self.capture_volume_changed.emit(capture_volume)

        if self._calibration_result is not None:
            self.calibration_result_updated.emit(self._calibration_result)
