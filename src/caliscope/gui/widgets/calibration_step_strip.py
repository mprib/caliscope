"""Compact workflow strip showing extrinsic calibration progress.

Displays Extract -> Calibrate -> Set origin as a single-line status strip.
Purely a status display, not a wizard — only the Extract step is clickable,
and only while it hasn't run yet.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from caliscope.core.workflow_status import StepStatus
from caliscope.gui.presenters.extrinsic_calibration_presenter import CalibrationStepData
from caliscope.gui.theme import Colors

_ICONS_DIR = Path(__file__).parent.parent / "icons"

_STATUS_ICON = {
    StepStatus.COMPLETE: ("status-complete.svg", Colors.SUCCESS),
    StepStatus.INCOMPLETE: ("status-incomplete.svg", Colors.WARNING),
    StepStatus.AVAILABLE: ("status-available.svg", Colors.PRIMARY),
    StepStatus.NOT_STARTED: ("status-not-started.svg", Colors.TEXT_MUTED),
}


def _load_colored_svg(icon: QSvgWidget, filename: str, color: str) -> None:
    svg_content = (_ICONS_DIR / filename).read_text()
    icon.load(QByteArray(svg_content.replace("currentColor", color).encode()))


class _LinkLabel(QLabel):
    """A QLabel styled and behaving like a hyperlink."""

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_underline(False)

    def _set_underline(self, underline: bool) -> None:
        decoration = "underline" if underline else "none"
        self.setStyleSheet(f"color: {Colors.PRIMARY}; font-size: 10px; text-decoration: {decoration};")

    def enterEvent(self, event) -> None:
        self._set_underline(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_underline(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _StepCell(QWidget):
    """A single step: status icon, bold name, and muted (or link) detail text."""

    link_clicked = Signal()

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._icon = QSvgWidget()
        self._icon.setFixedSize(16, 16)
        layout.addWidget(self._icon, alignment=Qt.AlignmentFlag.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(0)

        self._name_label = QLabel(name)
        self._name_label.setStyleSheet(f"font-weight: bold; color: {Colors.TEXT_PRIMARY}; font-size: 11px;")
        text_layout.addWidget(self._name_label)

        self._detail_label = QLabel()
        self._detail_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        text_layout.addWidget(self._detail_label)

        self._detail_link = _LinkLabel()
        self._detail_link.hide()
        self._detail_link.clicked.connect(self.link_clicked)
        text_layout.addWidget(self._detail_link)

        layout.addLayout(text_layout)

    def set_status(self, status: StepStatus, detail: str, *, as_link: bool = False) -> None:
        filename, color = _STATUS_ICON[status]
        _load_colored_svg(self._icon, filename, color)

        if as_link:
            self._detail_label.hide()
            self._detail_link.setText(detail)
            self._detail_link.show()
        else:
            self._detail_link.hide()
            self._detail_label.setText(detail)
            self._detail_label.show()


class CalibrationStepStrip(QWidget):
    """Compact horizontal strip: Extract -> Calibrate -> Set origin.

    Status display only. The Extract step becomes a clickable link to the
    Cameras tab while extraction hasn't been run yet; the other two steps
    are never clickable.
    """

    navigation_requested = Signal(str)  # Tab name to navigate to

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._extract_cell = _StepCell("Extract")
        self._extract_cell.link_clicked.connect(lambda: self.navigation_requested.emit("Cameras"))
        layout.addWidget(self._extract_cell, stretch=1)

        layout.addWidget(self._make_separator())

        self._calibrate_cell = _StepCell("Calibrate")
        layout.addWidget(self._calibrate_cell, stretch=1)

        layout.addWidget(self._make_separator())

        self._origin_cell = _StepCell("Set origin")
        layout.addWidget(self._origin_cell, stretch=1)

    def _make_separator(self) -> QLabel:
        separator = QLabel("▸")
        separator.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        return separator

    def set_data(self, data: CalibrationStepData) -> None:
        """Update all three step cells from presenter-computed status data."""
        extract_status, extract_detail = data.extract
        self._extract_cell.set_status(
            extract_status,
            extract_detail,
            as_link=extract_status == StepStatus.NOT_STARTED,
        )

        calibrate_status, calibrate_detail = data.calibrate
        self._calibrate_cell.set_status(calibrate_status, calibrate_detail)

        origin_status, origin_detail = data.origin
        self._origin_cell.set_status(origin_status, origin_detail)
