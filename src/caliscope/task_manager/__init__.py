"""Task management infrastructure for background operations.

TaskHandle and TaskManager are Qt-backed and load lazily on first access, so
importing this package (e.g. via headless calibration code that needs only
CancellationToken) does not require PySide6.
"""

from typing import TYPE_CHECKING

from caliscope.task_manager.cancellation import CancellationToken
from caliscope.task_manager.task_state import TaskState

if TYPE_CHECKING:
    from caliscope.task_manager.task_handle import TaskHandle
    from caliscope.task_manager.task_manager import TaskManager

__all__ = [
    "CancellationToken",
    "TaskHandle",
    "TaskManager",
    "TaskState",
]


def __getattr__(name: str):
    if name == "TaskHandle":
        from caliscope.task_manager.task_handle import TaskHandle

        return TaskHandle
    if name == "TaskManager":
        from caliscope.task_manager.task_manager import TaskManager

        return TaskManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
