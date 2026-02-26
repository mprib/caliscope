"""Dialog guiding the user to download ONNX model weights."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from caliscope.trackers.model_card import ModelCard


class ModelDownloadDialog(QDialog):
    """Modal dialog guiding the user to download ONNX model weights.

    Shows model metadata, download instructions, and provides buttons to
    copy the download URL and open the models folder.
    """

    def __init__(self, card: ModelCard, models_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._card = card
        self._models_dir = models_dir

        self.setWindowTitle("Model Download Required")
        self.setModal(True)
        self.setMinimumWidth(480)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel(f'The model "<b>{self._card.name}</b>" needs to be downloaded before it can be used.')
        header.setWordWrap(True)
        layout.addWidget(header)

        # Conditional metadata
        if self._card.file_size_mb is not None:
            size_label = QLabel(f"<b>File size:</b> ~{self._card.file_size_mb:.0f} MB")
            layout.addWidget(size_label)

        if self._card.license_info is not None:
            license_label = QLabel(f"<b>License:</b> {self._card.license_info}")
            license_label.setWordWrap(True)
            layout.addWidget(license_label)

        # Instructions
        steps = QLabel(
            "<b>Steps:</b><br>"
            '1. Click "Copy Download Link" below<br>'
            "2. Paste the link in your browser to download<br>"
            "3. Save the .onnx file to the models folder:"
        )
        steps.setWordWrap(True)
        layout.addWidget(steps)

        # Models folder path display
        path_layout = QHBoxLayout()
        path_label = QLabel(f"<code>{self._models_dir}</code>")
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        path_layout.addWidget(path_label, stretch=1)

        copy_path_btn = QPushButton("Copy Path")
        copy_path_btn.setFixedWidth(80)
        copy_path_btn.clicked.connect(self._copy_path)
        path_layout.addWidget(copy_path_btn)

        layout.addLayout(path_layout)

        layout.addSpacing(8)

        # Action buttons
        button_layout = QHBoxLayout()

        copy_link_btn = QPushButton("Copy Download Link")
        if self._card.source_url:
            copy_link_btn.clicked.connect(self._copy_download_link)
        else:
            copy_link_btn.setEnabled(False)
            copy_link_btn.setToolTip("No download URL configured")
        button_layout.addWidget(copy_link_btn)

        open_folder_btn = QPushButton("Open Models Folder")
        open_folder_btn.clicked.connect(self._open_models_folder)
        button_layout.addWidget(open_folder_btn)

        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

    def _copy_download_link(self) -> None:
        """Copy the model download URL to clipboard."""
        if self._card.source_url:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(self._card.source_url)

    def _copy_path(self) -> None:
        """Copy the models directory path to clipboard."""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(str(self._models_dir))

    def _open_models_folder(self) -> None:
        """Open the models directory in the system file manager."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._models_dir)))
