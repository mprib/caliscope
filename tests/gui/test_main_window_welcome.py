"""Integration tests for MainWindow welcome screen."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import rtoml

from caliscope.gui.main_widget import MainWindow
from caliscope.gui.widgets.welcome_widget import WelcomeWidget


def _make_settings(tmp_path: Path) -> Path:
    settings_path = tmp_path / "settings.toml"
    settings = {
        "recent_projects": [],
        "last_project_parent": str(tmp_path),
    }
    settings_path.write_text(rtoml.dumps(settings))
    return settings_path


def test_initial_central_widget_is_welcome(qapp, tmp_path):
    settings_path = _make_settings(tmp_path)
    with patch("caliscope.gui.main_widget.APP_SETTINGS_PATH", settings_path):
        window = MainWindow()
        assert isinstance(window.centralWidget(), WelcomeWidget)
        window.close()


def test_recent_projects_filters_nonexistent(qapp, tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    settings_path = tmp_path / "settings.toml"
    settings = {
        "recent_projects": [str(existing), str(tmp_path / "gone")],
        "last_project_parent": str(tmp_path),
    }
    settings_path.write_text(rtoml.dumps(settings))

    with patch("caliscope.gui.main_widget.APP_SETTINGS_PATH", settings_path):
        window = MainWindow()
        recents = window.recent_projects()
        assert str(existing) in recents
        assert str(tmp_path / "gone") not in recents
        window.close()


def test_sync_failure_shows_error(qapp, tmp_path):
    settings_path = _make_settings(tmp_path)
    workspace = tmp_path / "bad_project"
    workspace.mkdir()

    with (
        patch("caliscope.gui.main_widget.APP_SETTINGS_PATH", settings_path),
        patch(
            "caliscope.workspace_coordinator.WorkspaceCoordinator",
            side_effect=ValueError("corrupt settings"),
        ),
    ):
        window = MainWindow()
        window.launch_workspace(str(workspace))

        central = window.centralWidget()
        assert isinstance(central, WelcomeWidget)
        assert central._open_button.isEnabled()
        assert not central._status_label.isHidden()
        assert "corrupt settings" in central._status_label.text()
        window.close()


def test_worker_failure_shows_error(qapp, tmp_path):
    """Worker-phase failure (background thread) routes to welcome error state."""
    settings_path = _make_settings(tmp_path)
    workspace = tmp_path / "worker_fail_project"
    workspace.mkdir()

    gate = threading.Event()

    def failing_worker(token, handle):
        gate.wait(timeout=5)
        raise RuntimeError("corrupt bundle data")

    with patch("caliscope.gui.main_widget.APP_SETTINGS_PATH", settings_path):
        window = MainWindow()

        mock_coordinator = MagicMock()
        mock_coordinator.cleanup = MagicMock()

        from caliscope.task_manager import TaskManager

        task_manager = TaskManager()
        mock_coordinator.task_manager = task_manager
        handle = task_manager.submit(failing_worker, name="test_load", auto_start=False)

        def start_load(h):
            task_manager.start_task(h.task_id)

        mock_coordinator.start_load = start_load
        mock_coordinator.load_workspace = MagicMock(return_value=handle)

        with patch(
            "caliscope.workspace_coordinator.WorkspaceCoordinator",
            return_value=mock_coordinator,
        ):
            window.launch_workspace(str(workspace))

        received = threading.Event()
        handle.failed.connect(lambda *_: received.set())
        gate.set()

        for _ in range(100):
            qapp.processEvents()
            if received.wait(timeout=0.05):
                break

        central = window.centralWidget()
        assert isinstance(central, WelcomeWidget)
        assert central._open_button.isEnabled()
        assert "corrupt bundle data" in central._status_label.text()

        task_manager.shutdown()
        window.close()


def test_fail_then_retry_cleans_up_coordinator(qapp, tmp_path):
    """Second launch_workspace after a failure tears down the first coordinator."""
    settings_path = _make_settings(tmp_path)

    with patch("caliscope.gui.main_widget.APP_SETTINGS_PATH", settings_path):
        window = MainWindow()

        mock_coordinator = MagicMock()
        mock_coordinator.cleanup = MagicMock()
        window.coordinator = mock_coordinator

        with patch(
            "caliscope.workspace_coordinator.WorkspaceCoordinator",
            side_effect=ValueError("still broken"),
        ):
            window.launch_workspace(str(tmp_path / "attempt2"))

        mock_coordinator.cleanup.assert_called_once()
        assert not hasattr(window, "coordinator") or window.coordinator is not mock_coordinator
        window.close()


if __name__ == "__main__":
    import os
    import tempfile

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication([])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_initial_central_widget_is_welcome(app, tmp)
        print("PASS: test_initial_central_widget_is_welcome")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_recent_projects_filters_nonexistent(app, tmp)
        print("PASS: test_recent_projects_filters_nonexistent")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_sync_failure_shows_error(app, tmp)
        print("PASS: test_sync_failure_shows_error")
