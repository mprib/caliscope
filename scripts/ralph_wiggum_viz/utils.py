"""Core utilities for Ralph Wiggum GUI visual testing.

Provides reusable helpers for capturing screenshots and sequencing actions
in PySide6/Qt applications. These utilities support the "Visual Ralph Wiggum"
technique where Claude reviews screenshots to verify UI correctness.

Usage:
    from utils import capture_widget, process_events_for, schedule_actions

    app = QApplication(sys.argv)
    widget = MyWidget()
    widget.show()

    process_events_for(500)  # Let it render
    capture_widget(widget, "01_initial.png")
"""

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
from PySide6.QtWidgets import QWidget

OUTPUT_DIR = Path(__file__).parent / "output"


def ensure_output_dir() -> Path:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def capture_widget(widget: QWidget, filename: str) -> Path:
    """Capture a widget's current visual state to a PNG file.

    Uses widget.grab() which captures the widget as it appears on screen,
    including any child widgets and their current state.

    Args:
        widget: The Qt widget to capture
        filename: Output filename (should end in .png)

    Returns:
        Path to the saved screenshot
    """
    path = ensure_output_dir() / filename
    pixmap = widget.grab()
    pixmap.save(str(path))
    print(f"Captured: {path}")
    return path


def process_events_for(ms: int = 100) -> None:
    """Process Qt events for a specified duration.

    Allows the GUI to update, animations to run, and async operations
    to complete before capturing screenshots. Essential for ensuring
    the UI is in the expected state.

    Args:
        ms: Duration in milliseconds to process events
    """
    app = QCoreApplication.instance()
    if app is None:
        return

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def schedule_actions(actions: list[tuple[int, Callable[[], None]]]) -> None:
    """Schedule a sequence of actions with delays.

    Useful for scripting complex UI interactions where each step needs
    time for the UI to respond before the next action.

    Args:
        actions: List of (delay_ms, callable) tuples. Each callable is
            executed after its corresponding delay from when this
            function is called.

    Example:
        schedule_actions([
            (0, lambda: button.click()),
            (500, lambda: capture_widget(widget, "01_after_click.png")),
            (1000, lambda: app.quit()),
        ])
    """
    for delay_ms, action in actions:
        QTimer.singleShot(delay_ms, action)


def clear_output_dir() -> None:
    """Remove all files from the output directory.

    Useful at the start of a test run to ensure clean state.
    """
    output_dir = ensure_output_dir()
    for file in output_dir.iterdir():
        if file.is_file():
            file.unlink()
    print(f"Cleared: {output_dir}")
