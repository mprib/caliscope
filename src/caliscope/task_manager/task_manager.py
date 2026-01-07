"""Centralized manager for background task lifecycle."""

import logging
import uuid
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread

from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_state import TaskState

logger = logging.getLogger(__name__)

WorkerFn = Callable[[CancellationToken, TaskHandle], Any]


class _WorkerThread(QThread):
    """Proper QThread subclass - replaces the `thread.run = worker` pattern."""

    def __init__(
        self,
        worker_fn: WorkerFn,
        token: CancellationToken,
        handle: TaskHandle,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._worker_fn = worker_fn
        self._token = token
        self._handle = handle

    def run(self) -> None:
        """Execute worker function with exception handling."""
        self._handle._set_running()

        try:
            result = self._worker_fn(self._token, self._handle)

            if self._token.is_cancelled:
                self._handle._emit_cancelled()
            else:
                self._handle._emit_completed(result)

        except Exception as e:
            self._handle._emit_failed(type(e).__name__, str(e))


class TaskManager(QObject):
    """Centralized manager for background task lifecycle.

    Replaces the controller's pattern of storing threads in dictionaries
    without cleanup. Tracks active tasks and provides abort functionality.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tasks: dict[str, tuple[TaskHandle, _WorkerThread]] = {}

    def submit(
        self,
        worker: WorkerFn,
        name: str,
        task_id: str | None = None,
    ) -> TaskHandle:
        """Submit a task for background execution.

        Args:
            worker: Callable with signature (token, handle) -> Any
            name: Human-readable name for logging
            task_id: Optional identifier; auto-generated if not provided

        Returns:
            TaskHandle for monitoring/controlling the task
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        token = CancellationToken()
        handle = TaskHandle(task_id, name, token, parent=self)
        thread = _WorkerThread(worker, token, handle, parent=self)

        # Register cleanup on thread completion
        thread.finished.connect(lambda: self._cleanup_task(task_id))

        self._tasks[task_id] = (handle, thread)
        logger.info(f"Starting task '{name}' (id={task_id})")
        thread.start()

        return handle

    def cancel(self, task_id: str) -> bool:
        """Cancel a specific task. Returns True if found."""
        if task_id in self._tasks:
            handle, _ = self._tasks[task_id]
            handle.cancel()
            return True
        return False

    def cancel_all(self) -> int:
        """Cancel all running tasks. Returns count cancelled."""
        count = 0
        for handle, _ in self._tasks.values():
            if handle.state == TaskState.RUNNING:
                handle.cancel()
                count += 1
        logger.info(f"Cancelled {count} task(s)")
        return count

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Cancel all tasks and wait for completion.

        Called during application exit to ensure clean shutdown.
        """
        logger.info("TaskManager shutdown initiated")
        self.cancel_all()

        # Wait for each thread to finish
        per_thread_timeout = timeout_ms // max(len(self._tasks), 1)
        for task_id, (handle, thread) in list(self._tasks.items()):
            if thread.isRunning():
                logger.info(f"Waiting for task '{handle.name}' to finish...")
                finished = thread.wait(per_thread_timeout)
                if not finished:
                    logger.warning(f"Task '{handle.name}' did not finish in time")

        logger.info("TaskManager shutdown complete")

    def _cleanup_task(self, task_id: str) -> None:
        """Remove completed task from tracking dictionary."""
        if task_id in self._tasks:
            handle, _ = self._tasks.pop(task_id)
            logger.debug(f"Cleaned up task '{handle.name}' (id={task_id})")

    @property
    def running_tasks(self) -> list[TaskHandle]:
        """Get list of currently running tasks."""
        return [handle for handle, _ in self._tasks.values() if handle.state == TaskState.RUNNING]

    @property
    def active_task_count(self) -> int:
        """Number of tasks currently tracked (any state)."""
        return len(self._tasks)
