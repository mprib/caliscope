"""Keyboard-accessible navigation link for an existing local folder."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QKeyEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from caliscope.gui.theme import Colors
from caliscope.gui.widgets.link_label import LinkLabel


class FolderLink(LinkLabel):
    """Open an existing folder using the system file manager."""

    def __init__(self, text: str, folder: Path | None, parent: QWidget | None = None) -> None:
        self._folder: Path | None = None
        super().__init__(font_size_px=11, parent=parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setText(text)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.clicked.connect(self._open_folder)
        self.set_folder(folder)

    @property
    def folder(self) -> Path | None:
        """Current local destination, if one is available."""
        return self._folder

    def set_folder(self, folder: Path | None) -> None:
        """Set the destination without creating or otherwise changing it."""
        self._folder = folder
        available = folder is not None and folder.is_dir()
        self.setEnabled(available)
        self.setCursor(Qt.CursorShape.PointingHandCursor if available else Qt.CursorShape.ArrowCursor)
        if available:
            self.setToolTip(f"Open folder: {folder}")
        elif folder is None:
            self.setToolTip("Folder unavailable")
        else:
            self.setToolTip(f"Folder unavailable: {folder}")
        self._set_underline(False)

    def _set_underline(self, underline: bool) -> None:
        """Apply LinkLabel styling with a visible keyboard-focus indicator."""
        enabled = self.isEnabled()
        decoration = "underline" if underline and enabled else "none"
        color = Colors.PRIMARY if enabled else Colors.TEXT_DISABLED
        focus_border = f"1px solid {Colors.PRIMARY}" if enabled and self.hasFocus() else "1px solid transparent"
        self.setStyleSheet(
            f"color: {color}; font-size: {self._font_size_px}px; text-decoration: {decoration}; border: {focus_border};"
        )

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        self._set_underline(True)

    def focusOutEvent(self, event) -> None:  # type: ignore[override]
        super().focusOutEvent(event)
        self._set_underline(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._open_folder()
            event.accept()
            return
        super().keyPressEvent(event)

    def _open_folder(self) -> None:
        """Open the current target only while it remains an existing folder."""
        if not self.isEnabled() or self._folder is None or not self._folder.is_dir():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._folder)))
