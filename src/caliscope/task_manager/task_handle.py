"""Handle for monitoring and controlling a submitted task."""

import logging
from typing import Any

from PySide6.QtCore import QObject, Signal

from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_state import TaskState

logger = logging.getLogger(__name__)


class TaskHandle(QObject):
    """Handle for monitoring and controlling a submitted task.

    Created by TaskManager.submit() and returned to the caller.
    Workers receive this to report progress; callers connect to
    signals for status updates.
    """

    started = Signal()  # emitted when task begins running
    completed = Signal(object)  # result value from worker
    failed = Signal(str, str)  # exception_type, message
    cancelled = Signal()
    progress_updated = Signal(int, str)  # percent (0-100), message

    def __init__(
        self,
        task_id: str,
        name: str,
        token: CancellationToken,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._task_id = task_id
        self._name = name
        self._token = token
        self._state = TaskState.PENDING
        self._result: Any = None

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def result(self) -> Any:
        """Result from completed worker, or None."""
        return self._result

    def cancel(self) -> None:
        """Request this task be cancelled.

        Does not immediately stop the task - the worker must check
        token.is_cancelled cooperatively.
        """
        if self._state == TaskState.RUNNING:
            logger.info(f"Cancellation requested for task '{self._name}'")
            self._token.cancel()

    def report_progress(self, percent: int, message: str = "") -> None:
        """Report progress from within the worker."""
        self.progress_updated.emit(percent, message)

    # Internal methods called by _WorkerThread

    def _set_running(self) -> None:
        self._state = TaskState.RUNNING
        self.started.emit()

    def _emit_completed(self, result: Any) -> None:
        self._result = result
        self._state = TaskState.COMPLETED
        self.completed.emit(result)

    def _emit_failed(self, exc_type: str, message: str) -> None:
        self._state = TaskState.FAILED
        logger.error(f"Task '{self._name}' failed: {exc_type}: {message}")
        self.failed.emit(exc_type, message)

    def _emit_cancelled(self) -> None:
        self._state = TaskState.CANCELLED
        logger.info(f"Task '{self._name}' was cancelled")
        self.cancelled.emit()
