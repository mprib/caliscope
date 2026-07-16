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


def test_close_during_load_cancels_worker(qapp, tmp_path):
    """Closing the window mid-load cancels the running worker and never builds tabs."""
    settings_path = _make_settings(tmp_path)
    workspace = tmp_path / "closing_project"
    workspace.mkdir()

    gate = threading.Event()
    captured: dict[str, object] = {}

    def blocking_worker(token, handle):
        captured["token"] = token
        # Hold in RUNNING until cancelled; the gate is a safety release so the thread
        # can't outlive the test. Mirrors production's cooperative cancellation checkpoints.
        for _ in range(200):
            if token.is_cancelled:
                return
            if gate.wait(timeout=0.05):
                return

    with patch("caliscope.gui.main_widget.APP_SETTINGS_PATH", settings_path):
        window = MainWindow()

        from caliscope.task_manager import TaskManager
        from caliscope.task_manager.task_state import TaskState

        task_manager = TaskManager()
        handle = task_manager.submit(blocking_worker, name="test_load", auto_start=False)

        mock_coordinator = MagicMock()
        mock_coordinator.task_manager = task_manager
        mock_coordinator.load_workspace = MagicMock(return_value=handle)
        mock_coordinator.start_load = lambda h: task_manager.start_task(h.task_id)
        # Mirror the real coordinator: cleanup shuts the TaskManager down, cancelling tokens.
        mock_coordinator.cleanup = lambda: task_manager.shutdown(timeout_ms=2000)

        with patch(
            "caliscope.workspace_coordinator.WorkspaceCoordinator",
            return_value=mock_coordinator,
        ):
            window.launch_workspace(str(workspace))

        # Wait until the worker is actually running before closing.
        for _ in range(200):
            qapp.processEvents()
            if handle.state == TaskState.RUNNING and "token" in captured:
                break
        assert handle.state == TaskState.RUNNING

        window.close()  # closeEvent -> coordinator.cleanup() -> task_manager.shutdown()
        gate.set()  # safety net if the worker somehow outlived the cancel

        token = captured["token"]
        assert token is not None and token.is_cancelled  # type: ignore[union-attr]
        assert not hasattr(window, "central_tab")  # build_central_tabs never ran


def test_project_switch_swaps_welcome_and_cleans_coordinator(qapp, tmp_path):
    """Opening a new project while one is loaded tears down the old coordinator and,
    on sync failure, lands on the welcome error state (not stale, dead-coordinator tabs)."""
    from PySide6.QtWidgets import QTabWidget

    settings_path = _make_settings(tmp_path)

    with patch("caliscope.gui.main_widget.APP_SETTINGS_PATH", settings_path):
        window = MainWindow()

        # Simulate a loaded project: a coordinator plus a QTabWidget central widget
        # (what build_central_tabs installs).
        old_coordinator = MagicMock()
        old_coordinator.cleanup = MagicMock()
        window.coordinator = old_coordinator
        window.setCentralWidget(QTabWidget())

        old_gen = window._build_generation

        with patch(
            "caliscope.workspace_coordinator.WorkspaceCoordinator",
            side_effect=ValueError("corrupt settings"),
        ):
            window.launch_workspace(str(tmp_path / "second_project"))

        old_coordinator.cleanup.assert_called_once()

        central = window.centralWidget()
        assert isinstance(central, WelcomeWidget)
        assert not central._status_label.isHidden()
        assert "corrupt settings" in central._status_label.text()

        # A stale deferred-build tick from the previous generation is a no-op:
        # it must not raise and must not touch the (now deleted) coordinator.
        window._build_next_deferred_tab(old_gen)

        window.close()


if __name__ == "__main__":
    import os
    import tempfile

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    tests = [
        test_initial_central_widget_is_welcome,
        test_recent_projects_filters_nonexistent,
        test_sync_failure_shows_error,
        test_worker_failure_shows_error,
        test_fail_then_retry_cleans_up_coordinator,
        test_close_during_load_cancels_worker,
        test_project_switch_swaps_welcome_and_cleans_coordinator,
    ]
    for test in tests:
        with tempfile.TemporaryDirectory() as td:
            test(app, Path(td))
        print(f"PASS: {test.__name__}")
