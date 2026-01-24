"""Tests for ExplorerPresenter."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from caliscope.synthetic.explorer.presenter import (
    ExplorerPresenter,
    PipelineResult,
)
from caliscope.synthetic.scene_factories import quick_test_scene
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.trajectory import Trajectory
from caliscope.task_manager.task_manager import TaskManager


@pytest.fixture
def qapp():
    """Ensure QCoreApplication exists for Qt signal tests."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


@pytest.fixture
def task_manager(qapp) -> TaskManager:
    """Create TaskManager for tests."""
    return TaskManager()


@pytest.fixture
def presenter(task_manager: TaskManager) -> ExplorerPresenter:
    """Create ExplorerPresenter with default scenario."""
    return ExplorerPresenter(task_manager)


def test_construction_with_default_scene(presenter: ExplorerPresenter) -> None:
    """Presenter initializes with default ring scene."""
    assert presenter.scene is not None
    assert presenter.scene.n_cameras == 4  # Default ring has 4 cameras
    assert presenter.n_frames == 20  # Default orbital n_frames


def test_set_scene_replaces_scene_and_emits(presenter: ExplorerPresenter) -> None:
    """set_scene() replaces scene and emits signal."""
    # Create a different scene with 5 frames
    camera_array = CameraSynthesizer().add_ring(n=4, radius_mm=2000.0, height_mm=500.0).build()
    calibration_object = CalibrationObject.planar_grid(rows=3, cols=4, spacing_mm=50.0)
    trajectory = Trajectory.stationary(n_frames=5)

    new_scene = SyntheticScene(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
    )

    signal_received = []
    presenter.scene_changed.connect(lambda s: signal_received.append(s))

    presenter.set_scene(new_scene)

    assert presenter.n_frames == 5
    assert len(signal_received) == 1
    assert signal_received[0] is new_scene


def test_is_linkage_killed_normalizes_order(presenter: ExplorerPresenter) -> None:
    """is_linkage_killed() works regardless of argument order."""
    # Set up killed linkage via FilterConfig (not via removed kill_linkage method)
    presenter._filter_config = presenter._filter_config.with_killed_linkage(2, 3)

    assert presenter.is_linkage_killed(2, 3)
    assert presenter.is_linkage_killed(3, 2)  # Order doesn't matter


def test_set_frame_clamps_and_emits(presenter: ExplorerPresenter) -> None:
    """set_frame() clamps to valid range and emits signal."""
    n_frames = presenter.n_frames

    signal_received = []
    presenter.frame_changed.connect(lambda idx: signal_received.append(idx))

    # Set to valid frame
    presenter.set_frame(5)
    assert presenter.current_frame == 5
    assert signal_received[-1] == 5

    # Set beyond max - should clamp
    presenter.set_frame(1000)
    assert presenter.current_frame == n_frames - 1

    # Set below min - should clamp
    presenter.set_frame(-5)
    assert presenter.current_frame == 0

    # Should have received 3 signals
    assert len(signal_received) == 3


def test_run_pipeline_executes_stages(presenter: ExplorerPresenter) -> None:
    """run_pipeline() executes bootstrap, optimize, and align stages.

    Uses a shorter scene to speed up the test.
    """
    # Use quick_test_scene for faster execution
    presenter.set_scene(quick_test_scene())

    # Wait for pipeline to complete using QEventLoop
    loop = QEventLoop()
    result_holder = []

    def on_finished(result: PipelineResult) -> None:
        result_holder.append(result)
        loop.quit()

    def on_failed(msg: str) -> None:
        loop.quit()
        pytest.fail(f"Pipeline failed: {msg}")

    presenter.pipeline_finished.connect(on_finished)
    presenter.pipeline_failed.connect(on_failed)

    # Start pipeline
    presenter.run_pipeline()

    # Set timeout (30 seconds)
    QTimer.singleShot(30000, loop.quit)

    # Wait for completion
    loop.exec()

    # Check result
    assert len(result_holder) == 1, "Pipeline did not complete (timeout or failure)"
    result = result_holder[0]

    assert result.ground_truth_cameras is not None
    assert result.ground_truth_world_points is not None

    # Should have successfully bootstrapped
    assert result.bootstrapped_cameras is not None
    assert result.bootstrapped_world_points is not None
    assert result.bootstrap_error is None

    # Should have successfully optimized
    assert result.optimized_cameras is not None
    assert result.optimized_world_points is not None
    assert result.optimization_error is None

    # Should have successfully aligned
    assert result.aligned_cameras is not None
    assert result.aligned_world_points is not None
    assert result.alignment_error is None


def test_coverage_matrix_reflects_filter(presenter: ExplorerPresenter) -> None:
    """Coverage matrix updates when filter changes."""
    # Get initial coverage
    initial_coverage = presenter.coverage_matrix
    assert initial_coverage is not None

    # Kill linkage between cameras 0 and 1 via FilterConfig
    presenter._filter_config = presenter._filter_config.with_killed_linkage(0, 1)
    presenter._apply_filter()

    # Coverage should change
    new_coverage = presenter.coverage_matrix
    assert new_coverage is not None

    # Off-diagonal element [0, 1] should be reduced (or zero)
    assert new_coverage[0, 1] < initial_coverage[0, 1]


def test_pipeline_failure_emits_signal(presenter: ExplorerPresenter) -> None:
    """Pipeline failure emits pipeline_failed signal."""
    # Create a minimal scene that will fail bootstrap
    # (not enough shared observations between cameras when linkages are killed)
    camera_array = CameraSynthesizer().add_ring(n=4, radius_mm=2000.0, height_mm=500.0).build()
    calibration_object = CalibrationObject.planar_grid(rows=2, cols=2, spacing_mm=50.0)
    trajectory = Trajectory.stationary(n_frames=1)

    sparse_scene = SyntheticScene(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
    )
    presenter.set_scene(sparse_scene)

    # Kill all linkages to guarantee failure via FilterConfig
    presenter._filter_config = (
        presenter._filter_config.with_killed_linkage(0, 1)
        .with_killed_linkage(0, 2)
        .with_killed_linkage(0, 3)
        .with_killed_linkage(1, 2)
        .with_killed_linkage(1, 3)
        .with_killed_linkage(2, 3)
    )
    presenter._apply_filter()

    # Wait for pipeline to complete or fail
    loop = QEventLoop()
    result_holder = []
    error_holder = []

    def on_finished(result: PipelineResult) -> None:
        result_holder.append(result)
        loop.quit()

    def on_failed(msg: str) -> None:
        error_holder.append(msg)
        loop.quit()

    presenter.pipeline_finished.connect(on_finished)
    presenter.pipeline_failed.connect(on_failed)

    # Start pipeline
    presenter.run_pipeline()

    # Set timeout (10 seconds)
    QTimer.singleShot(10000, loop.quit)

    # Wait for completion
    loop.exec()

    # Either failed signal or finished with error
    if error_holder:
        assert isinstance(error_holder[0], str)
    else:
        assert len(result_holder) == 1
        result = result_holder[0]
        # Should have bootstrap error since no linkages exist
        assert result.bootstrap_error is not None


if __name__ == "__main__":
    """Debug harness for manual testing and inspection."""
    from pathlib import Path

    import sys
    from PySide6.QtWidgets import QApplication

    # Setup debug output directory
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    task_manager = TaskManager()
    presenter = ExplorerPresenter(task_manager)

    print(f"Initial scene: {presenter.scene.n_cameras} cameras, {presenter.n_frames} frames")
    coverage = presenter.coverage_matrix
    assert coverage is not None
    print(f"Coverage matrix shape: {coverage.shape}")

    # Kill a linkage via FilterConfig and observe change
    presenter._filter_config = presenter._filter_config.with_killed_linkage(0, 1)
    presenter._apply_filter()
    print("After killing (0,1) linkage:")
    print(f"  Killed linkages: {presenter.filter_config.killed_linkages}")
    new_coverage = presenter.coverage_matrix
    assert new_coverage is not None
    print(f"  Coverage[0,1]: {new_coverage[0, 1]}")

    print("\nTo run pipeline test, uncomment the section below and run with 'python test_explorer_presenter.py'")

    # Uncomment to test pipeline execution
    # def on_finished(result):
    #     print(f"\nPipeline finished!")
    #     print(f"  Bootstrap error: {result.bootstrap_error}")
    #     print(f"  Optimization error: {result.optimization_error}")
    #     print(f"  Alignment error: {result.alignment_error}")
    #     if result.optimized_cameras:
    #         print(f"  Optimized {len(result.optimized_cameras.cameras)} cameras")
    #     app.quit()
    #
    # def on_failed(msg):
    #     print(f"\nPipeline failed: {msg}")
    #     app.quit()
    #
    # presenter.pipeline_finished.connect(on_finished)
    # presenter.pipeline_failed.connect(on_failed)
    # presenter.run_pipeline()
    #
    # sys.exit(app.exec())
