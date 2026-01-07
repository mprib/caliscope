"""Task lifecycle states."""

from enum import Enum, auto


class TaskState(Enum):
    """Lifecycle states for managed tasks."""

    PENDING = auto()  # Submitted but not yet started
    RUNNING = auto()  # Currently executing
    COMPLETED = auto()  # Finished successfully
    FAILED = auto()  # Raised an exception
    CANCELLED = auto()  # Cancelled before or during execution
