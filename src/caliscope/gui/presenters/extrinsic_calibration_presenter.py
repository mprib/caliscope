"""Presenter for extrinsic calibration workflow.

Manages the workflow from ImagePoints through bundle adjustment optimization
to calibrated camera poses. Wraps PointDataBundle operations with Qt signals
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

from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.build_paired_pose_network import (
    build_paired_pose_network,
)
from caliscope.core.charuco import Charuco
from caliscope.core.point_data import ImagePoints
from caliscope.core.point_data_bundle import PointDataBundle
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

    NEEDS_BOOTSTRAP = auto()  # Have ImagePoints path, need to triangulate
    NEEDS_OPTIMIZATION = auto()  # Have WorldPoints, not yet optimized
    OPTIMIZING = auto()  # Background optimization running
    CALIBRATED = auto()  # Optimization complete, can refine


@dataclass(frozen=True)
class FilterPreviewData:
    """Data for filter UI showing translation between modes.

    Maps percentile-to-remove to corresponding pixel threshold.
    E.g., {5: 1.23} means "removing worst 5% would remove observations
    with error > 1.23 pixels".
    """

    total_observations: int
    mean_error: float
    # Maps percentile-to-remove → pixel threshold
    threshold_at_percentile: dict[int, float]

    @classmethod
    def empty(cls) -> FilterPreviewData:
        """Create empty preview data."""
        return cls(total_observations=0, mean_error=0.0, threshold_at_percentile={})


@dataclass(frozen=True)
class QualityPanelData:
    """Quality metrics for display in the UI.

    Contains reprojection error statistics and optimization metadata.
    """

    # Reprojection metrics
    overall_rmse: float
    n_observations: int
    n_world_points: int

    # Per-camera table rows: (port, n_obs, rmse)
    camera_rows: list[tuple[int, int, float]]

    # Optimization metadata
    converged: bool
    iterations: int

    # Filter preview for UI
    filter_preview: FilterPreviewData | None


class ExtrinsicCalibrationPresenter(QObject):
    """Presenter for extrinsic calibration workflow.

    Manages the extraction of camera extrinsics from charuco observations.
    Coordinates bootstrap triangulation, bundle adjustment optimization,
    and coordinate frame transformations.

    Signals:
        state_changed: Emitted when computed state changes. View updates UI.
        progress_updated: Emitted during optimization with (percent, message).
        quality_updated: Emitted when metrics refresh after calibration.
        calibration_complete: Emitted when optimization finishes.
            Contains the optimized PointDataBundle.
        view_model_updated: Emitted when 3D view needs refresh.
            Contains PlaybackViewModel.

    Usage:
        presenter = ExtrinsicCalibrationPresenter(
            task_manager, camera_array, image_points_path, charuco
        )
        presenter.run_calibration()  # Bootstrap + optimize
        # On completion: calibration_complete emitted with bundle
    """

    # State signals
    state_changed = Signal(object)  # ExtrinsicCalibrationState

    # Progress signals
    progress_updated = Signal(int, str)  # (percent, message)

    # Result signals
    quality_updated = Signal(object)  # QualityPanelData
    calibration_complete = Signal(object)  # PointDataBundle
    view_model_updated = Signal(object)  # PlaybackViewModel

    def __init__(
        self,
        task_manager: TaskManager,
        camera_array: CameraArray,
        image_points_path: Path,
        charuco: Charuco,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            task_manager: TaskManager for background processing
            camera_array: Initial camera configuration (extrinsics may be unset)
            image_points_path: Path to image_points.csv from Phase 3
            charuco: Charuco board definition for alignment
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._task_manager = task_manager
        self._camera_array = camera_array
        self._image_points_path = image_points_path
        self._charuco = charuco

        # Processing state (managed internally)
        self._bundle: PointDataBundle | None = None
        self._task_handle: TaskHandle | None = None

        # View state
        self._current_sync_index: int = 0

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
            return ExtrinsicCalibrationState.OPTIMIZING

        if self._bundle is not None and self._bundle.optimization_status is not None:
            return ExtrinsicCalibrationState.CALIBRATED

        if self._bundle is not None:
            return ExtrinsicCalibrationState.NEEDS_OPTIMIZATION

        return ExtrinsicCalibrationState.NEEDS_BOOTSTRAP

    @property
    def bundle(self) -> PointDataBundle | None:
        """Current bundle (None before calibration)."""
        return self._bundle

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
        then runs bundle adjustment optimization. Emits calibration_complete
        with the optimized bundle.

        Only valid in NEEDS_BOOTSTRAP state.
        """
        if self.state != ExtrinsicCalibrationState.NEEDS_BOOTSTRAP:
            logger.warning(f"Cannot run calibration in state {self.state}")
            return

        # Capture for closure - deepcopy camera_array since bootstrap mutates it
        image_points_path = self._image_points_path
        camera_array = deepcopy(self._camera_array)

        def worker(token: CancellationToken, handle: TaskHandle) -> PointDataBundle:
            return self._execute_calibration(image_points_path, camera_array, token, handle)

        self._task_handle = self._task_manager.submit(
            worker,
            name="Extrinsic calibration",
        )
        self._task_handle.completed.connect(self._on_calibration_complete)
        self._task_handle.failed.connect(self._on_calibration_failed)
        self._task_handle.cancelled.connect(self._on_calibration_cancelled)

        self._emit_state_changed()

    def _execute_calibration(
        self,
        image_points_path: Path,
        camera_array: CameraArray,
        token: CancellationToken,
        handle: TaskHandle,
    ) -> PointDataBundle:
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

        handle.report_progress(40, "Building bundle")
        bundle = PointDataBundle(camera_array, image_points, world_points)

        if token.is_cancelled:
            raise InterruptedError("Calibration cancelled")

        handle.report_progress(50, "Running initial optimization")
        optimized = bundle.optimize(ftol=1e-8, verbose=0)
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

    def re_optimize(self) -> None:
        """Run optimization again on current bundle.

        Useful after filtering or if previous optimization was interrupted.
        Only valid in CALIBRATED or NEEDS_OPTIMIZATION states.
        """
        if self._bundle is None:
            logger.warning("Cannot re-optimize: no bundle available")
            return

        if self._is_task_active():
            logger.warning("Cannot re-optimize: task already running")
            return

        # Capture for closure
        bundle = self._bundle

        def worker(token: CancellationToken, handle: TaskHandle) -> PointDataBundle:
            handle.report_progress(10, "Running optimization")
            optimized = bundle.optimize(ftol=1e-8, verbose=0)
            handle.report_progress(100, "Complete")
            logger.info(f"Re-optimization RMSE: {optimized.reprojection_report.overall_rmse:.3f}px")
            return optimized

        self._task_handle = self._task_manager.submit(
            worker,
            name="Re-optimize bundle",
        )
        self._task_handle.completed.connect(self._on_calibration_complete)
        self._task_handle.failed.connect(self._on_calibration_failed)
        self._task_handle.cancelled.connect(self._on_calibration_cancelled)

        self._emit_state_changed()

    # -------------------------------------------------------------------------
    # Filtering Operations (Implemented in 4.3)
    # -------------------------------------------------------------------------

    def filter_by_percentile(self, percentile: float) -> None:
        """Filter worst N% of observations and re-optimize.

        Args:
            percentile: Percentage of worst observations to remove (0-100)
        """
        raise NotImplementedError("Implemented in Subphase 4.3")

    def filter_by_threshold(self, max_error_pixels: float) -> None:
        """Filter observations above threshold and re-optimize.

        Args:
            max_error_pixels: Maximum reprojection error in pixels to keep
        """
        raise NotImplementedError("Implemented in Subphase 4.3")

    def get_filter_preview(self) -> FilterPreviewData:
        """Get error stats for filter UI.

        Returns data allowing the View to show translation between filter modes.
        E.g., "Removing 5% would remove observations with error > 1.23px"
        """
        raise NotImplementedError("Implemented in Subphase 4.3")

    # -------------------------------------------------------------------------
    # Coordinate Frame Operations (Implemented in 4.3)
    # -------------------------------------------------------------------------

    def rotate(self, axis: str, degrees: float) -> None:
        """Rotate coordinate frame around axis.

        Args:
            axis: "x", "y", or "z"
            degrees: Rotation angle in degrees (positive = counter-clockwise)
        """
        raise NotImplementedError("Implemented in Subphase 4.3")

    def align_to_origin(self, sync_index: int) -> None:
        """Set world origin to board position at sync_index.

        Args:
            sync_index: Frame index where board position defines origin
        """
        raise NotImplementedError("Implemented in Subphase 4.3")

    # -------------------------------------------------------------------------
    # View Control (Implemented in 4.3)
    # -------------------------------------------------------------------------

    def set_sync_index(self, index: int) -> None:
        """Update current frame for 3D view.

        Args:
            index: Sync index to display
        """
        raise NotImplementedError("Implemented in Subphase 4.3")

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

    def _on_calibration_complete(self, bundle: PointDataBundle) -> None:
        """Handle successful calibration/optimization completion."""
        logger.info(f"Calibration complete. RMSE: {bundle.reprojection_report.overall_rmse:.3f}px")

        self._bundle = bundle
        self._task_handle = None

        # Set initial sync index to first available frame
        sync_indices = bundle.unique_sync_indices
        if len(sync_indices) > 0:
            self._current_sync_index = int(sync_indices[0])

        self._emit_state_changed()
        self._emit_quality_updated()
        self.calibration_complete.emit(bundle)

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

    def _emit_quality_updated(self) -> None:
        """Build and emit quality panel data from current bundle."""
        if self._bundle is None:
            return

        report = self._bundle.reprojection_report
        status = self._bundle.optimization_status

        # Build per-camera rows: (port, n_obs, rmse)
        camera_rows: list[tuple[int, int, float]] = []
        for port in sorted(report.by_camera.keys()):
            n_obs = int((self._bundle.image_points.df["port"] == port).sum())
            rmse = report.by_camera[port]
            camera_rows.append((port, n_obs, rmse))

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
