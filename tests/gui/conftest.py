"""Shared fixtures for GUI tests."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from PySide6.QtCore import QObject

from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle


class FakeTaskManager(QObject):
    """TaskManager stand-in that never runs workers.

    submit() returns a real TaskHandle that stays PENDING until the test
    finishes it with complete/fail/cancel. Those drive the handle through the
    same methods the worker thread uses, so presenters see real state and real
    signals. Presenters connect with QueuedConnection, so call
    qapp.processEvents() (or a qtbot wait) before asserting.
    """

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[tuple[TaskHandle, Any]] = []
        self.started: list[str] = []

    def submit(self, worker, name: str, task_id: str | None = None, auto_start: bool = False) -> TaskHandle:
        handle = TaskHandle(task_id or str(uuid.uuid4()), name, CancellationToken(), parent=self)
        self.submitted.append((handle, worker))
        if auto_start:
            self.started.append(handle.task_id)
        return handle

    def start_task(self, task_id: str) -> bool:
        self.started.append(task_id)
        return True

    def cancel_all(self) -> int:
        return 0

    def shutdown(self, timeout_ms: int = 5000) -> None:
        pass

    def handles(self, name: str) -> list[TaskHandle]:
        return [handle for handle, _ in self.submitted if handle.name == name]

    def worker(self, handle: TaskHandle):
        return next(worker for h, worker in self.submitted if h is handle)

    def run(self, handle: TaskHandle) -> None:
        """Run the worker inline and finish the handle with its outcome."""
        handle._set_running()
        try:
            result = self.worker(handle)(handle._token, handle)
        except Exception as exc:
            handle._emit_failed(type(exc).__name__, str(exc))
        else:
            handle._emit_completed(result)

    @staticmethod
    def start(handle: TaskHandle) -> None:
        handle._set_running()

    @staticmethod
    def complete(handle: TaskHandle, result: Any = None) -> None:
        handle._emit_completed(result)

    @staticmethod
    def fail(handle: TaskHandle, exc_type: str = "RuntimeError", message: str = "boom") -> None:
        handle._emit_failed(exc_type, message)

    @staticmethod
    def cancel(handle: TaskHandle) -> None:
        handle._emit_cancelled()


@pytest.fixture
def fake_task_manager(qapp) -> FakeTaskManager:
    return FakeTaskManager()
