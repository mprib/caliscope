"""Presenter for Synthetic Calibration Explorer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.core.point_data_bundle import PointDataBundle
from caliscope.core.bootstrap_pose.build_paired_pose_network import (
    build_paired_pose_network,
)
from caliscope.synthetic.scene import SyntheticScene
from caliscope.synthetic.scenario_config import ScenarioConfig, default_ring_scenario
from caliscope.synthetic.filter_config import FilterConfig
from caliscope.synthetic.coverage import compute_coverage_matrix
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.cancellation import CancellationToken

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Stages in the calibration pipeline."""

    GROUND_TRUTH = auto()
    BOOTSTRAPPED = auto()
    OPTIMIZED = auto()
    ALIGNED = auto()


@dataclass
class PipelineResult:
    """Results from running the calibration pipeline."""

    ground_truth_cameras: CameraArray
    ground_truth_world_points: WorldPoints

    bootstrapped_cameras: CameraArray | None = None
    bootstrapped_world_points: WorldPoints | None = None

    optimized_cameras: CameraArray | None = None
    optimized_world_points: WorldPoints | None = None

    aligned_cameras: CameraArray | None = None
    aligned_world_points: WorldPoints | None = None

    bootstrap_error: str | None = None
    optimization_error: str | None = None
    alignment_error: str | None = None

    # Error metrics (computed after alignment)
    reprojection_rmse: float | None = None
    camera_metrics: tuple[CameraMetrics, ...] = ()


@dataclass
class CameraMetrics:
    """Per-camera error metrics."""

    port: int
    rotation_error_deg: float
    translation_error_mm: float
    reprojection_rmse: float
    n_observations: int


def _compute_pose_error(
    estimated_rotation: np.ndarray,
    estimated_translation: np.ndarray,
    ground_truth_rotation: np.ndarray,
    ground_truth_translation: np.ndarray,
) -> tuple[float, float]:
    """Compute rotation error (degrees) and translation error (mm).

    Rotation error uses geodesic distance on SO(3) via Rodrigues.
    Translation error is Euclidean distance between camera positions.
    """
    # Rotation error: geodesic distance on SO(3)
    R_rel = estimated_rotation @ ground_truth_rotation.T
    # OpenCV stubs incorrectly infer matrix multiplication result type
    rodrigues, _ = cv2.Rodrigues(R_rel)  # type: ignore[arg-type]
    rotation_rad = np.linalg.norm(rodrigues)
    rotation_deg = float(np.degrees(rotation_rad))

    # Translation error: Euclidean distance between camera positions
    pos_est = -estimated_rotation.T @ estimated_translation
    pos_gt = -ground_truth_rotation.T @ ground_truth_translation
    translation_mm = float(np.linalg.norm(pos_est - pos_gt))

    return rotation_deg, translation_mm


class ExplorerPresenter(QObject):
    """Presenter for Synthetic Calibration Explorer.

    Manages the lifecycle of:
    - Scenario configuration (rig, trajectory, object)
    - Filter configuration (killed linkages, dropped cameras)
    - Pipeline execution (bootstrap, optimize, align)
    - Result comparison (ground truth vs optimized)

    Signals:
        scene_changed: Emitted when scenario config changes. Contains new SyntheticScene.
        filter_changed: Emitted when filter config changes. Contains new coverage matrix.
        pipeline_started: Emitted when pipeline execution begins.
        pipeline_stage_complete: Emitted after each pipeline stage. Contains PipelineStage.
        pipeline_finished: Emitted when pipeline completes. Contains PipelineResult.
        pipeline_failed: Emitted on pipeline error. Contains error message.
        frame_changed: Emitted when current frame changes. Contains sync_index.
    """

    scene_changed = Signal(object)  # SyntheticScene
    filter_changed = Signal(object)  # NDArray coverage matrix
    pipeline_started = Signal()
    pipeline_stage_complete = Signal(object)  # PipelineStage
    pipeline_finished = Signal(object)  # PipelineResult
    pipeline_failed = Signal(str)
    frame_changed = Signal(int)

    def __init__(
        self,
        task_manager: TaskManager,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self._task_manager = task_manager
        self._pipeline_task: TaskHandle | None = None

        # State
        self._config = default_ring_scenario()
        self._scene: SyntheticScene | None = None
        self._filter_config = FilterConfig()
        self._filtered_image_points: ImagePoints | None = None
        self._result: PipelineResult | None = None
        self._current_frame: int = 0

        # Initialize scene
        self._rebuild_scene()

    @property
    def config(self) -> ScenarioConfig:
        """Current scenario configuration."""
        return self._config

    @property
    def scene(self) -> SyntheticScene | None:
        """Current synthetic scene (None if not built)."""
        return self._scene

    @property
    def filter_config(self) -> FilterConfig:
        """Current filter configuration."""
        return self._filter_config

    @property
    def coverage_matrix(self) -> NDArray[np.int64] | None:
        """Current coverage matrix (after filtering)."""
        if self._filtered_image_points is None or self._scene is None:
            return None
        return compute_coverage_matrix(self._filtered_image_points, self._scene.n_cameras)

    @property
    def result(self) -> PipelineResult | None:
        """Most recent pipeline result."""
        return self._result

    @property
    def current_frame(self) -> int:
        """Current frame index for visualization."""
        return self._current_frame

    @property
    def n_frames(self) -> int:
        """Number of frames in current scene."""
        return self._scene.n_frames if self._scene else 0

    # --- Configuration Methods ---

    def set_config(self, config: ScenarioConfig) -> None:
        """Replace entire scenario configuration and rebuild scene."""
        self._config = config
        self._filter_config = config.filter_config
        self._result = None
        self._rebuild_scene()

    def update_rig(
        self,
        rig_type: Literal["ring", "linear", "nested_rings"],
        rig_params: dict[str, Any],
    ) -> None:
        """Update camera rig configuration."""
        self._config = ScenarioConfig(
            rig_type=rig_type,
            rig_params=rig_params,
            trajectory_type=self._config.trajectory_type,
            trajectory_params=self._config.trajectory_params,
            object_type=self._config.object_type,
            object_params=self._config.object_params,
            pixel_noise_sigma=self._config.pixel_noise_sigma,
            filter_config=self._filter_config,
            random_seed=self._config.random_seed,
            name=self._config.name,
            description=self._config.description,
        )
        self._result = None
        self._rebuild_scene()

    def update_trajectory(
        self,
        traj_type: Literal["orbital", "linear", "stationary"],
        traj_params: dict[str, Any],
    ) -> None:
        """Update trajectory configuration."""
        self._config = ScenarioConfig(
            rig_type=self._config.rig_type,
            rig_params=self._config.rig_params,
            trajectory_type=traj_type,
            trajectory_params=traj_params,
            object_type=self._config.object_type,
            object_params=self._config.object_params,
            pixel_noise_sigma=self._config.pixel_noise_sigma,
            filter_config=self._filter_config,
            random_seed=self._config.random_seed,
            name=self._config.name,
            description=self._config.description,
        )
        self._result = None
        self._rebuild_scene()

    def update_noise(self, pixel_sigma: float, seed: int) -> None:
        """Update noise configuration."""
        self._config = ScenarioConfig(
            rig_type=self._config.rig_type,
            rig_params=self._config.rig_params,
            trajectory_type=self._config.trajectory_type,
            trajectory_params=self._config.trajectory_params,
            object_type=self._config.object_type,
            object_params=self._config.object_params,
            pixel_noise_sigma=pixel_sigma,
            filter_config=self._filter_config,
            random_seed=seed,
            name=self._config.name,
            description=self._config.description,
        )
        self._result = None
        self._rebuild_scene()

    # --- Filter Methods ---

    def is_linkage_killed(self, cam_a: int, cam_b: int) -> bool:
        """Check if linkage is currently killed."""
        normalized = (min(cam_a, cam_b), max(cam_a, cam_b))
        return normalized in self._filter_config.killed_linkages

    # --- Frame Navigation ---

    def set_frame(self, frame: int) -> None:
        """Set current frame for visualization."""
        if self._scene is None:
            return

        frame = max(0, min(frame, self._scene.n_frames - 1))
        if frame != self._current_frame:
            self._current_frame = frame
            self.frame_changed.emit(frame)

    # --- Pipeline Execution ---

    def run_pipeline(self) -> None:
        """Run the full calibration pipeline in a background thread.

        Stages:
        1. Bootstrap: Stereo calibration -> PairedPoseNetwork -> initial extrinsics
        2. Optimize: Bundle adjustment via PointDataBundle.optimize()
        3. Align: Umeyama alignment to ground truth using obj_loc at origin_frame
        """
        if self._scene is None:
            self.pipeline_failed.emit("No scene configured")
            return

        if self._filtered_image_points is None:
            self.pipeline_failed.emit("No filtered image points available")
            return

        if self._pipeline_task is not None:
            self.pipeline_failed.emit("Pipeline already running")
            return

        # Capture current state for closure
        scene = self._scene
        filtered_points = self._filtered_image_points

        def pipeline_worker(token: CancellationToken, handle: TaskHandle) -> PipelineResult:
            return self._execute_pipeline(scene, filtered_points, token)

        self._pipeline_task = self._task_manager.submit(
            pipeline_worker,
            name="Synthetic Pipeline",
        )
        self._pipeline_task.completed.connect(self._on_pipeline_complete)
        self._pipeline_task.failed.connect(self._on_pipeline_failed)

        self.pipeline_started.emit()

    def _execute_pipeline(
        self,
        scene: SyntheticScene,
        image_points: ImagePoints,
        token: CancellationToken,
    ) -> PipelineResult:
        """Execute pipeline stages. Runs in background thread."""
        result = PipelineResult(
            ground_truth_cameras=scene.camera_array,
            ground_truth_world_points=scene.world_points,
        )

        # Stage 1: Bootstrap
        try:
            intrinsics_only = scene.intrinsics_only_cameras()

            # Build pose network from stereo pairs
            pose_network = build_paired_pose_network(
                image_points,
                intrinsics_only,
                method="pnp",
            )

            # Apply to get initial extrinsics
            pose_network.apply_to(intrinsics_only)

            # Triangulate initial world points
            bootstrapped_world = image_points.triangulate(intrinsics_only)

            result.bootstrapped_cameras = intrinsics_only
            result.bootstrapped_world_points = bootstrapped_world

            logger.info("Bootstrap complete")
            # Note: Can't emit signals from background thread - result will be
            # processed when completed signal fires

        except Exception as e:
            result.bootstrap_error = str(e)
            logger.error(f"Bootstrap failed: {e}")
            return result

        if token.is_cancelled:
            return result

        # Stage 2: Optimize
        try:
            bundle = PointDataBundle(
                camera_array=result.bootstrapped_cameras,
                image_points=image_points,
                world_points=result.bootstrapped_world_points,
            )

            optimized_bundle = bundle.optimize(ftol=1e-8, verbose=0)

            result.optimized_cameras = optimized_bundle.camera_array
            result.optimized_world_points = optimized_bundle.world_points

            logger.info(f"Optimization complete. RMSE: {optimized_bundle.reprojection_report.overall_rmse:.3f}px")

        except Exception as e:
            result.optimization_error = str(e)
            logger.error(f"Optimization failed: {e}")
            return result

        if token.is_cancelled:
            return result

        # Stage 3: Align
        try:
            origin_frame = scene.trajectory.origin_frame

            aligned_bundle = PointDataBundle(
                camera_array=result.optimized_cameras,
                image_points=image_points,
                world_points=result.optimized_world_points,
            ).align_to_object(sync_index=origin_frame)

            result.aligned_cameras = aligned_bundle.camera_array
            result.aligned_world_points = aligned_bundle.world_points

            logger.info("Alignment complete")

            # Compute error metrics after successful alignment
            reprojection_report = optimized_bundle.reprojection_report
            result.reprojection_rmse = reprojection_report.overall_rmse

            # Compute per-camera pose errors and observation counts
            camera_metrics_list = []
            for port in result.aligned_cameras.cameras:
                cam_result = result.aligned_cameras.cameras[port]
                cam_gt = result.ground_truth_cameras.cameras[port]

                # After alignment, all cameras have pose data
                assert cam_result.rotation is not None
                assert cam_result.translation is not None
                assert cam_gt.rotation is not None
                assert cam_gt.translation is not None

                rotation_error, translation_error = _compute_pose_error(
                    cam_result.rotation,
                    cam_result.translation,
                    cam_gt.rotation,
                    cam_gt.translation,
                )

                # Count observations for this camera
                n_obs = int((image_points.df["port"] == port).sum())

                # Get per-camera reprojection RMSE
                camera_reproj_rmse = reprojection_report.by_camera.get(port, 0.0)

                camera_metrics_list.append(
                    CameraMetrics(
                        port=port,
                        rotation_error_deg=rotation_error,
                        translation_error_mm=translation_error,
                        reprojection_rmse=camera_reproj_rmse,
                        n_observations=n_obs,
                    )
                )

            result.camera_metrics = tuple(camera_metrics_list)

        except Exception as e:
            result.alignment_error = str(e)
            logger.error(f"Alignment failed: {e}")

        return result

    def _on_pipeline_complete(self, result: PipelineResult) -> None:
        """Handle pipeline completion."""
        self._result = result
        self._pipeline_task = None
        self.pipeline_finished.emit(result)

    def _on_pipeline_failed(self, exc_type: str, message: str) -> None:
        """Handle pipeline failure."""
        self._pipeline_task = None
        self.pipeline_failed.emit(f"{exc_type}: {message}")

    def cancel_pipeline(self) -> None:
        """Cancel running pipeline."""
        if self._pipeline_task is not None:
            self._pipeline_task.cancel()

    # --- Internal Methods ---

    def _rebuild_scene(self) -> None:
        """Rebuild scene from current config."""
        try:
            self._scene = self._config.build_scene()
            self._current_frame = 0
            self._apply_filter()
            self.scene_changed.emit(self._scene)

        except Exception as e:
            logger.error(f"Failed to build scene: {e}")
            self._scene = None

    def _apply_filter(self) -> None:
        """Apply current filter to scene and emit coverage update."""
        if self._scene is None:
            return

        self._filtered_image_points = self._filter_config.apply(self._scene.image_points_noisy)

        coverage = compute_coverage_matrix(self._filtered_image_points, self._scene.n_cameras)
        self.filter_changed.emit(coverage)
