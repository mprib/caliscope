"""Tests for TaskManager.

Uses mock workers to test lifecycle without real operations.
Requires Qt (PySide6) but no xvfb - uses QCoreApplication only.
"""

import threading
import time

import pytest
from PySide6.QtCore import QCoreApplication

from caliscope.task_manager import (
    TaskHandle,
    TaskManager,
    TaskState,
)


@pytest.fixture(scope="module")
def qapp():
    """Create QCoreApplication for signal/slot system."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


def _wait_for_condition(condition_fn, timeout: float, qapp):
    """Poll condition while processing Qt events."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        qapp.processEvents()
        if condition_fn():
            return
        time.sleep(0.01)
    raise TimeoutError(f"Condition not met within {timeout}s")


def test_submit_returns_handle(qapp):
    manager = TaskManager()

    def worker(token, handle):
        return 42

    handle = manager.submit(worker, "test_task")

    assert isinstance(handle, TaskHandle)
    assert handle.name == "test_task"
    assert handle.task_id is not None

    manager.shutdown(timeout_ms=1000)


def test_worker_result_emitted_via_completed_signal(qapp):
    manager = TaskManager()
    received = {}
    start_event = threading.Event()

    def worker(token, handle):
        start_event.wait()  # Wait until signal is connected
        return "result_value"

    handle = manager.submit(worker, "test_task")
    handle.completed.connect(lambda r: received.update({"result": r}))
    qapp.processEvents()  # Ensure connection is processed
    start_event.set()  # Now let worker proceed

    _wait_for_condition(lambda: "result" in received, timeout=2.0, qapp=qapp)

    assert received.get("result") == "result_value"
    assert handle.state == TaskState.COMPLETED

    manager.shutdown(timeout_ms=1000)


def test_worker_exception_emitted_via_failed_signal(qapp):
    manager = TaskManager()
    received = {}
    start_event = threading.Event()

    def worker(token, handle):
        start_event.wait()  # Wait until signal is connected
        raise ValueError("test error")

    handle = manager.submit(worker, "failing_task")
    handle.failed.connect(lambda t, m: received.update({"type": t, "msg": m}))
    qapp.processEvents()  # Ensure connection is processed
    start_event.set()  # Now let worker proceed

    _wait_for_condition(lambda: "type" in received, timeout=2.0, qapp=qapp)

    assert received.get("type") == "ValueError"
    assert "test error" in received.get("msg", "")
    assert handle.state == TaskState.FAILED

    manager.shutdown(timeout_ms=1000)


def test_cancel_emits_cancelled_signal(qapp):
    manager = TaskManager()
    received = {"cancelled": False}

    def worker(token, handle):
        while not token.is_cancelled:
            token.sleep_unless_cancelled(0.1)

    handle = manager.submit(worker, "cancellable_task")
    handle.cancelled.connect(lambda: received.update({"cancelled": True}))

    time.sleep(0.05)  # Let worker start
    manager.cancel(handle.task_id)

    _wait_for_condition(lambda: received["cancelled"], timeout=2.0, qapp=qapp)

    assert handle.state == TaskState.CANCELLED

    manager.shutdown(timeout_ms=1000)


def test_progress_updates_emitted(qapp):
    manager = TaskManager()
    progress_reports = []
    start_event = threading.Event()

    def worker(token, handle):
        start_event.wait()  # Wait until signal is connected
        for i in range(3):
            handle.report_progress(i * 33, f"Step {i}")
            time.sleep(0.01)

    handle = manager.submit(worker, "progress_task")
    handle.progress_updated.connect(lambda p, m: progress_reports.append((p, m)))
    qapp.processEvents()  # Ensure connection is processed
    start_event.set()  # Now let worker proceed

    _wait_for_condition(lambda: len(progress_reports) >= 3, timeout=2.0, qapp=qapp)

    assert len(progress_reports) >= 3
    assert progress_reports[0] == (0, "Step 0")

    manager.shutdown(timeout_ms=1000)


def test_shutdown_cancels_and_waits(qapp):
    manager = TaskManager()
    completed = {"done": False}

    def worker(token, handle):
        # Simulates a task that respects cancellation
        for _ in range(10):
            if token.is_cancelled:
                return "cancelled"
            time.sleep(0.05)
        completed["done"] = True
        return "finished"

    # Submit multiple tasks
    handles = [manager.submit(worker, f"task_{i}") for i in range(3)]

    time.sleep(0.05)  # Let workers start
    manager.shutdown(timeout_ms=5000)

    # All tasks should have finished or been cancelled
    for handle in handles:
        assert handle.state in (TaskState.COMPLETED, TaskState.CANCELLED)


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])

    # test_submit_returns_handle(app)
    # test_worker_result_emitted_via_completed_signal(app)
    # test_worker_exception_emitted_via_failed_signal(app)
    # test_cancel_emits_cancelled_signal(app)
    test_progress_updates_emitted(app)
    # test_shutdown_cancels_and_waits(app)

    print("All TaskManager tests passed!")
