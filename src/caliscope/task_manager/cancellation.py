"""Cooperative cancellation for background tasks."""

from threading import Event


class CancellationToken:
    """Cooperative cancellation for background tasks.

    Wraps threading.Event with inverted semantics: Event.set() means
    "cancelled", which allows wait() to wake up immediately on cancellation.
    """

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        """Check if abort was requested."""
        return self._event.is_set()

    def sleep_unless_cancelled(self, seconds: float) -> bool:
        """Sleep for duration, returning early if cancelled.

        Returns True if cancelled, False if timeout elapsed normally.
        """
        return self._event.wait(timeout=seconds)

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()
