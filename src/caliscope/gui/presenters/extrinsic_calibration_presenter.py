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
from typing import Literal

import numpy as np
from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.build_paired_pose_network import (
    build_paired_pose_network,
)
from caliscope.core.charuco import Charuco
from caliscope.core.coverage_analysis import compute_coverage_matrix
from caliscope.core.point_data import ImagePoints
from caliscope.core.point_data_bundle import PointDataBundle
from caliscope.core.scale_accuracy import compute_scale_accuracy
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

    Provides bidirectional preview:
    - threshold_at_percentile: percentile-to-remove → pixel threshold
    - errors: raw error array for computing percentile at any threshold
    """

    total_observations: int
    mean_error: float
    # Maps percentile-to-remove → pixel threshold
    threshold_at_percentile: dict[int, float]
    # Raw errors for computing reverse lookup (threshold → percentile)
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
    scale_accuracy_updated = Signal(object)  # ScaleAccuracyData
    coverage_updated = Signal(object, object)  # (coverage_matrix, port_labels)
    calibration_complete = Signal(object)  # PointDataBundle
    view_model_updated = Signal(object)  # PlaybackViewModel

    def __init__(
        self,
        task_manager: TaskManager,
        camera_array: CameraArray,
        image_points_path: Path,
        charuco: Charuco,
        existing_bundle: PointDataBundle | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            task_manager: TaskManager for background processing
            camera_array: Initial camera configuration (extrinsics may be unset)
            image_points_path: Path to image_points.csv from Phase 3
            charuco: Charuco board definition for alignment
            existing_bundle: Pre-loaded PointDataBundle for restoring calibrated state.
                If provided, presenter starts in CALIBRATED state with visualization ready.
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._task_manager = task_manager
        self._camera_array = camera_array
        self._image_points_path = image_points_path
        self._charuco = charuco

        # Processing state (managed internally)
        self._bundle: PointDataBundle | None = existing_bundle
        self._task_handle: TaskHandle | None = None

        # Pre-loaded image points for initial coverage display
        self._initial_image_points: ImagePoints | None = None

        # View state
        self._current_sync_index: int = 0

        # Scale accuracy reference frame (set by align_to_origin)
        self._reference_sync_index: int | None = None

        # Load image points for coverage display (from bundle if available, else CSV)
        if existing_bundle is not None:
            self._initial_image_points = existing_bundle.image_points
            # Set initial sync index from bundle
            sync_indices = existing_bundle.unique_sync_indices
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
        self._task_handle.progress_updated.connect(self.progress_updated)

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
        self._task_handle.progress_updated.connect(self.progress_updated)

        self._emit_state_changed()

    # -------------------------------------------------------------------------
    # Filtering Operations (Implemented in 4.3)
    # -------------------------------------------------------------------------

    def filter_by_percentile(self, percentile: float) -> None:
        """Filter worst N% of observations and re-optimize.

        Args:
            percentile: Percentage of worst observations to remove (0-100)
        """
        if self._bundle is None:
            return

        filtered = self._bundle.filter_by_percentile_error(percentile)
        logger.info(f"Filtered {percentile}% worst observations, {len(filtered.image_points.df)} remaining")
        self._submit_optimization(filtered)

    def filter_by_threshold(self, max_error_pixels: float) -> None:
        """Filter observations above threshold and re-optimize.

        Args:
            max_error_pixels: Maximum reprojection error in pixels to keep
        """
        if self._bundle is None:
            return

        filtered = self._bundle.filter_by_absolute_error(max_error_pixels)
        logger.info(f"Filtered to error < {max_error_pixels}px, {len(filtered.image_points.df)} remaining")
        self._submit_optimization(filtered)

    def get_filter_preview(self) -> FilterPreviewData:
        """Get error stats for filter UI.

        Returns data allowing the View to show translation between filter modes:
        - Percentile mode: "Removing 5% would remove observations > 1.23px"
        - Absolute mode: "Removing observations > 1.0px would filter 3.2%"
        """
        if self._bundle is None:
            return FilterPreviewData.empty()

        report = self._bundle.reprojection_report
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
        if self._bundle is None:
            return

        # PointDataBundle.rotate() expects Literal["x", "y", "z"]
        # The domain method validates the axis value
        axis_typed: Literal["x", "y", "z"] = axis  # type: ignore[assignment]
        new_bundle = self._bundle.rotate(axis_typed, degrees)
        logger.info(f"Rotated coordinate frame {degrees}° around {axis}-axis")
        self._update_bundle(new_bundle)

    def align_to_origin(self, sync_index: int) -> None:
        """Set world origin to board position at sync_index.

        Also computes and emits scale accuracy metrics by comparing
        triangulated world points to known object geometry.

        Args:
            sync_index: Frame index where board position defines origin
        """
        if self._bundle is None:
            return

        new_bundle = self._bundle.align_to_object(sync_index)
        logger.info(f"Aligned world origin to object at sync_index={sync_index}")

        self._reference_sync_index = sync_index
        self._update_bundle(new_bundle)
        self._emit_scale_accuracy()

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
        if self._bundle is None:
            return

        # Clamp to valid range
        sync_indices = self._bundle.unique_sync_indices
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
        self._emit_coverage_updated()
        self._emit_view_model_updated()
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

    def _emit_view_model_updated(self) -> None:
        """Build and emit PlaybackViewModel for 3D visualization."""
        if self._bundle is None:
            return

        # Import here to avoid circular dependency at module level
        from caliscope.ui.viz.playback_view_model import PlaybackViewModel

        view_model = PlaybackViewModel(
            camera_array=self._bundle.camera_array,
            world_points=self._bundle.world_points,
        )
        self.view_model_updated.emit(view_model)

    def _emit_scale_accuracy(self) -> None:
        """Compute and emit scale accuracy metrics.

        Compares triangulated world points at the reference frame to their
        known ground truth positions from the tracker's object geometry.
        Works with any rigid tracker that provides obj_loc_* columns.
        """
        if self._bundle is None or self._reference_sync_index is None:
            return

        sync_index = self._reference_sync_index
        world_df = self._bundle.world_points.df
        img_df = self._bundle.image_points.df

        # Get world points at reference frame
        world_at_ref = world_df[world_df["sync_index"] == sync_index]
        if world_at_ref.empty:
            logger.warning(f"No world points at reference sync_index {sync_index}")
            return

        # Get image points with object locations at reference frame
        img_at_ref = img_df[img_df["sync_index"] == sync_index]
        if img_at_ref.empty:
            logger.warning(f"No image points at reference sync_index {sync_index}")
            return

        # Match by point_id to get corresponding world/object point pairs
        # Use drop_duplicates on img_at_ref since multiple cameras may see same point_id
        obj_points_df = img_at_ref[["point_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]].drop_duplicates(
            subset=["point_id"]
        )

        merged = world_at_ref.merge(obj_points_df, on="point_id", how="inner")

        if len(merged) < 2:
            logger.warning(f"Insufficient matched points for scale accuracy: {len(merged)}")
            return

        # Handle planar objects (z=0 or NaN)
        if merged["obj_loc_z"].isna().all():
            merged = merged.copy()
            merged["obj_loc_z"] = 0.0

        # Filter out any remaining NaN values
        valid_mask = ~merged[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].isna().any(axis=1)
        merged = merged[valid_mask]

        if len(merged) < 2:
            logger.warning("Insufficient valid points after NaN filtering")
            return

        # Extract arrays for scale accuracy computation
        world_points = merged[["x_coord", "y_coord", "z_coord"]].to_numpy()
        object_points = merged[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].to_numpy()

        try:
            scale_data = compute_scale_accuracy(world_points, object_points, sync_index)
            logger.info(
                f"Scale accuracy at frame {sync_index}: "
                f"RMSE={scale_data.distance_rmse_mm:.2f}mm, "
                f"relative={scale_data.relative_error_percent:.2f}%"
            )
            self.scale_accuracy_updated.emit(scale_data)
        except ValueError as e:
            logger.warning(f"Could not compute scale accuracy: {e}")

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
        - Quality data and view model (if existing bundle loaded)

        This enables the view to show the correct initial state:
        - Fresh start: coverage heatmap, "Calibrate" button
        - Restored session: 3D visualization, quality metrics, "Re-optimize" button
        """
        # Always emit coverage from initial image points
        self._emit_initial_coverage()

        # If we have an existing bundle, emit quality and view model for 3D viz
        if self._bundle is not None:
            logger.info("Emitting initial state from existing bundle")
            self._emit_quality_updated()
            self._emit_coverage_updated()  # Use bundle's coverage (may differ after filtering)
            self._emit_view_model_updated()

    def emit_initial_coverage(self) -> None:
        """Deprecated: Use emit_initial_state() instead."""
        self.emit_initial_state()

    def _emit_initial_coverage(self) -> None:
        """Emit coverage matrix from pre-loaded ImagePoints.

        Internal method - use emit_initial_state() from the view.
        Uses ports discovered from ImagePoints data (not posed cameras).
        """
        if self._initial_image_points is None:
            return

        df = self._initial_image_points.df
        if len(df) == 0:
            return

        # Build port-to-index mapping from actual data
        actual_ports = sorted(df["port"].unique())
        port_to_index = {int(port): idx for idx, port in enumerate(actual_ports)}

        coverage = compute_coverage_matrix(self._initial_image_points, port_to_index)
        labels = [f"C{p}" for p in actual_ports]

        self.coverage_updated.emit(coverage, labels)

    def _emit_coverage_updated(self) -> None:
        """Emit coverage matrix data for heatmap visualization.

        Computes pairwise observation counts between all posed cameras.
        Labels use port numbers (C1, C2, etc.) matching the camera array.
        """
        if self._bundle is None:
            return

        camera_array = self._bundle.camera_array
        port_to_index = camera_array.posed_port_to_index

        if not port_to_index:
            logger.debug("No posed cameras for coverage matrix")
            return

        coverage = compute_coverage_matrix(self._bundle.image_points, port_to_index)
        labels = [f"C{p}" for p in sorted(port_to_index.keys())]

        self.coverage_updated.emit(coverage, labels)

    def _submit_optimization(self, bundle: PointDataBundle) -> None:
        """Submit bundle optimization as background task.

        Used by filter methods to avoid duplicating task setup code.
        After filtering, the bundle needs re-optimization to update
        camera extrinsics and world points.

        Args:
            bundle: Filtered bundle to optimize
        """
        if self._is_task_active():
            logger.warning("Cannot start optimization: task already running")
            return

        def worker(token: CancellationToken, handle: TaskHandle) -> PointDataBundle:
            handle.report_progress(10, "Running optimization")
            optimized = bundle.optimize(ftol=1e-8, verbose=0)
            handle.report_progress(100, "Complete")
            logger.info(f"Post-filter optimization RMSE: {optimized.reprojection_report.overall_rmse:.3f}px")
            return optimized

        self._task_handle = self._task_manager.submit(worker, name="Optimize bundle")
        self._task_handle.completed.connect(self._on_calibration_complete)
        self._task_handle.failed.connect(self._on_calibration_failed)
        self._task_handle.cancelled.connect(self._on_calibration_cancelled)
        self._task_handle.progress_updated.connect(self.progress_updated)
        self._emit_state_changed()

    def _update_bundle(self, bundle: PointDataBundle) -> None:
        """Update internal bundle and emit signals.

        Called after synchronous bundle-modifying operations (rotate, align).
        These operations don't change the optimization status, just transform
        the coordinate frame.

        Args:
            bundle: New bundle after transformation
        """
        self._bundle = bundle
        self._emit_quality_updated()
        self._emit_view_model_updated()
        self.calibration_complete.emit(bundle)
