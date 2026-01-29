"""Widget for displaying structural warnings with severity-appropriate styling."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from caliscope.core.coverage_analysis import StructuralWarning, WarningSeverity


class StructuralWarningsWidget(QWidget):
    """Display structural warnings with severity-appropriate styling.

    Shows a list of warnings from coverage analysis, color-coded by severity:
    - CRITICAL: Red, bold (calibration will fail)
    - WARNING: Orange (may cause issues)
    - INFO: Gray (informational)

    When no warnings exist, displays a green "No structural issues detected" message.
    """

    STYLES = {
        WarningSeverity.CRITICAL: "color: #FF4444; font-weight: bold;",
        WarningSeverity.WARNING: "color: #FFA500;",
        WarningSeverity.INFO: "color: #888888;",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._labels: list[QLabel] = []

    def set_warnings(self, warnings: list[StructuralWarning]) -> None:
        """Update displayed warnings.

        Args:
            warnings: List of structural warnings to display.
                      Shows 'No issues' message if empty.
        """
        # Clear existing labels
        for label in self._labels:
            label.deleteLater()
        self._labels.clear()

        if not warnings:
            label = QLabel("\u2713 No structural issues detected")  # ✓ check mark
            label.setStyleSheet("color: #4CAF50;")  # Green
            self._layout.addWidget(label)
            self._labels.append(label)
        else:
            for warning in warnings:
                label = QLabel(warning.message)
                label.setStyleSheet(self.STYLES[warning.severity])
                label.setWordWrap(True)
                self._layout.addWidget(label)
                self._labels.append(label)

    def clear(self) -> None:
        """Clear all warnings."""
        for label in self._labels:
            label.deleteLater()
        self._labels.clear()
