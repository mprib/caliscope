"""Reusable hyperlink-styled QLabel with click signal."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QWidget

from caliscope.gui.theme import Colors


class LinkLabel(QLabel):
    """A QLabel styled and behaving like a hyperlink."""

    clicked = Signal()

    def __init__(self, font_size_px: int = 10, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._font_size_px = font_size_px
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_underline(False)

    def _set_underline(self, underline: bool) -> None:
        decoration = "underline" if underline else "none"
        self.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-size: {self._font_size_px}px; text-decoration: {decoration};"
        )

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._set_underline(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._set_underline(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
