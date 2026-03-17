"""GC confinement for Qt3D / shiboken6 thread safety.

Problem (PYSIDE-810): shiboken6 Python wrappers around Qt C++ objects are not
thread-safe for cyclic GC traversal. Python's cyclic collector can be triggered
from any thread (e.g., a worker thread that allocates enough objects to cross
the collection threshold). When that traversal visits a shiboken wrapper whose
underlying C++ object is simultaneously being accessed by Qt3D's render thread,
the result is a use-after-free crash.

Solution: disable automatic cyclic collection and run it only on the main thread
via a QTimer. Reference counting is unaffected — only cyclic collection is
confined. The interval default (10 s) is generous; cyclic garbage in a GUI
application is rare and short-lived, so the delay has no observable impact.

Usage:
    gc_timer = enable()      # after QApplication(), before any Qt3D widgets
    app.exec()
    disable(gc_timer)        # after event loop exits
"""

import gc
import logging

from PySide6.QtCore import QTimer

logger = logging.getLogger(__name__)


def _collect() -> None:
    """Run a full cyclic GC pass on the main thread."""
    collected = gc.collect()
    if collected:
        logger.debug("GC collected %d objects", collected)


def enable(interval_ms: int = 10_000) -> QTimer:
    """Disable automatic cyclic GC and replace it with main-thread-only collection.

    Must be called after QApplication is constructed and before any Qt3D widgets
    are created. Returns the QTimer so the caller can hold a reference and pass it
    to disable() later.
    """
    gc.disable()
    timer = QTimer()
    timer.setInterval(interval_ms)
    timer.timeout.connect(_collect)
    timer.start()
    logger.debug("GC confinement enabled (interval=%d ms)", interval_ms)
    return timer


def disable(timer: QTimer) -> None:
    """Stop the confinement timer and restore automatic cyclic GC.

    Runs a final gc.collect() before handing control back to Python's automatic
    collector, ensuring no cyclic garbage is left over from the session.
    """
    timer.stop()
    gc.enable()
    gc.collect()
    logger.debug("GC confinement disabled")
