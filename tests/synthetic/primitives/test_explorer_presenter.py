"""Tests for ExplorerPresenter."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from caliscope.synthetic.explorer.presenter import (
    ExplorerPresenter,
    PipelineResult,
)
from caliscope.synthetic.scenario_config import ScenarioConfig
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


def test_construction_with_default_scenario(presenter: ExplorerPresenter) -> None:
    """Presenter initializes with default ring scenario."""
    assert presenter.scene is not None
    assert presenter.config.rig_type == "ring"
    assert presenter.n_frames == 20  # Default orbital n_frames


def test_set_config_rebuilds_scene(presenter: ExplorerPresenter) -> None:
    """set_config() rebuilds scene and emits signal."""
    # Create a different config
    sparse_config = ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0, "height_mm": 500.0},
        trajectory_type="stationary",
        trajectory_params={"n_frames": 5},
        object_type="planar_grid",
        object_params={"rows": 3, "cols": 4, "spacing_mm": 50.0},
        name="Test Scenario",
    )

    signal_received = []
    presenter.scene_changed.connect(lambda s: signal_received.append(s))

    presenter.set_config(sparse_config)

    assert presenter.config.name == "Test Scenario"
    assert presenter.n_frames == 5
    assert len(signal_received) == 1


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

    Uses a shorter scenario to speed up the test.
    """
    # Use smaller scenario for faster execution
    short_config = ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0, "height_mm": 500.0},
        trajectory_type="orbital",
        trajectory_params={
            "n_frames": 5,  # Reduced from default 20
            "radius_mm": 200.0,
            "arc_extent_deg": 180.0,  # Half orbit
            "tumble_rate": 0.5,
        },
        object_type="planar_grid",
        object_params={"rows": 3, "cols": 4, "spacing_mm": 50.0},  # Smaller grid
        name="Quick Test",
    )
    presenter.set_config(short_config)

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
    # Create an invalid scenario that will fail bootstrap
    # (not enough shared observations between cameras)
    bad_config = ScenarioConfig(
        rig_type="ring",
        rig_params={"n_cameras": 4, "radius_mm": 2000.0, "height_mm": 500.0},
        trajectory_type="stationary",
        trajectory_params={"n_frames": 1},  # Very few frames
        object_type="planar_grid",
        object_params={"rows": 2, "cols": 2, "spacing_mm": 50.0},  # Very few points
        name="Sparse Scenario",
    )
    presenter.set_config(bad_config)

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
    print(f"Coverage matrix shape: {presenter.coverage_matrix.shape}")

    # Kill a linkage via FilterConfig and observe change
    presenter._filter_config = presenter._filter_config.with_killed_linkage(0, 1)
    presenter._apply_filter()
    print("After killing (0,1) linkage:")
    print(f"  Killed linkages: {presenter.filter_config.killed_linkages}")
    print(f"  Coverage[0,1]: {presenter.coverage_matrix[0, 1]}")

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
