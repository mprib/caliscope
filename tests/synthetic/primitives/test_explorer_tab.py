"""Tests for ExplorerTab widget."""

import pytest
from PySide6.QtWidgets import QApplication

from caliscope.synthetic.explorer import ExplorerTab
from caliscope.task_manager.task_manager import TaskManager


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def task_manager(qapp):
    """Create TaskManager for tests."""
    manager = TaskManager()
    yield manager
    manager.shutdown()


class TestExplorerTabConstruction:
    """Test basic construction and component creation."""

    def test_construction_creates_all_components(self, task_manager):
        """Tab should create storyboard, heatmap, and controls."""
        tab = ExplorerTab(task_manager)
        assert tab._storyboard is not None
        assert tab._heatmap is not None
        assert tab._preset_combo is not None
        assert tab._run_button is not None
        assert tab._frame_slider is not None
        assert tab._status_label is not None
        tab.cleanup()

    def test_default_preset_is_selected(self, task_manager):
        """First preset should be selected on construction."""
        tab = ExplorerTab(task_manager)
        assert tab._preset_combo.currentIndex() == 0
        assert tab._presenter.scene is not None
        tab.cleanup()


class TestPresetChange:
    """Test preset dropdown behavior."""

    def test_preset_change_updates_scene(self, task_manager):
        """Changing preset should rebuild the scene."""
        tab = ExplorerTab(task_manager)
        initial_scene = tab._presenter.scene
        tab._preset_combo.setCurrentIndex(1)
        new_scene = tab._presenter.scene
        assert new_scene is not None
        assert new_scene is not initial_scene
        tab.cleanup()


class TestRunButton:
    """Test run button behavior."""

    def test_run_button_disables_during_pipeline(self, task_manager):
        """Run button should disable while pipeline is running."""
        tab = ExplorerTab(task_manager)
        assert tab._run_button.isEnabled()
        tab._run_button.click()
        assert not tab._run_button.isEnabled()
        assert tab._run_button.text() == "Running..."
        tab.cleanup()


class TestCleanup:
    """Test lifecycle management."""

    def test_cleanup_stops_pipeline(self, task_manager):
        """Cleanup should cancel any running pipeline."""
        tab = ExplorerTab(task_manager)
        tab._run_button.click()
        tab.cleanup()

    def test_vtk_suspend_resume(self, task_manager):
        """VTK suspend/resume should not raise."""
        tab = ExplorerTab(task_manager)
        tab.suspend_vtk()
        tab.resume_vtk()
        tab.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
