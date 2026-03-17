"""Dialog guiding the user to download ONNX model weights."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from caliscope.task_manager.task_manager import TaskManager
from caliscope.trackers.model_card import ModelCard


class ModelDownloadDialog(QDialog):
    """Modal dialog for downloading ONNX model weights.

    Three-phase lifecycle:
    1. Confirmation: Shows model info, license, checkbox to accept terms
    2. Downloading: Progress bar with cancel button
    3a. Success: Brief confirmation then auto-close
    3b. Failure: Error message with fallback manual instructions
    3c. Cancellation: Revert to Phase 1 for retry
    """

    def __init__(
        self,
        card: ModelCard,
        models_dir: Path,
        task_manager: TaskManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._card = card
        self._models_dir = models_dir
        self._task_manager = task_manager
        self._task_handle = None

        self.setWindowTitle("Model Download Required")
        self.setModal(True)
        self.setMinimumWidth(520)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog layout."""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(16, 16, 16, 16)
        self._main_layout.setSpacing(12)

        # Header - visible in all phases
        self._header = QLabel(f'The model "<b>{self._card.name}</b>" needs to be downloaded.')
        self._header.setWordWrap(True)
        self._main_layout.addWidget(self._header)

        # Phase 1: Confirmation controls
        self._confirmation_widget = self._create_confirmation_widget()
        self._main_layout.addWidget(self._confirmation_widget)

        # Phase 2: Download progress controls (initially hidden)
        self._download_widget = self._create_download_widget()
        self._download_widget.hide()
        self._main_layout.addWidget(self._download_widget)

        # Phase 3: Result controls (initially hidden)
        self._result_widget = self._create_result_widget()
        self._result_widget.hide()
        self._main_layout.addWidget(self._result_widget)

    def _create_confirmation_widget(self) -> QWidget:
        """Create Phase 1 confirmation controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # File size (if available)
        if self._card.file_size_mb is not None:
            size_label = QLabel(f"<b>File size:</b> ~{self._card.file_size_mb:.0f} MB")
            layout.addWidget(size_label)

        # License info with clickable link (if available)
        if self._card.license_info is not None:
            license_text = f"<b>License:</b> {self._card.license_info}"
            if self._card.license_url:
                license_text += f' (<a href="{self._card.license_url}">view full license</a>)'
            license_label = QLabel(license_text)
            license_label.setWordWrap(True)
            license_label.setOpenExternalLinks(True)
            layout.addWidget(license_label)

        layout.addSpacing(8)

        # License acceptance checkbox
        self._accept_checkbox = QCheckBox("I have reviewed and accept the license terms")
        self._accept_checkbox.stateChanged.connect(self._on_checkbox_changed)
        layout.addWidget(self._accept_checkbox)

        layout.addSpacing(8)

        # Buttons: Download, Open Models Folder, Cancel
        button_layout = QHBoxLayout()

        self._download_btn = QPushButton("Download")
        self._download_btn.setEnabled(False)  # Disabled until checkbox checked
        self._download_btn.clicked.connect(self._start_download)
        button_layout.addWidget(self._download_btn)

        open_folder_btn = QPushButton("Open Models Folder")
        open_folder_btn.clicked.connect(self._open_models_folder)
        button_layout.addWidget(open_folder_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

        return widget

    def _create_download_widget(self) -> QWidget:
        """Create Phase 2 download progress controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        # Status label
        self._status_label = QLabel("Downloading...")
        layout.addWidget(self._status_label)

        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._cancel_download_btn = QPushButton("Cancel")
        self._cancel_download_btn.clicked.connect(self._cancel_download)
        button_layout.addWidget(self._cancel_download_btn)

        layout.addLayout(button_layout)

        return widget

    def _create_result_widget(self) -> QWidget:
        """Create Phase 3 result controls (success or failure)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Result message label
        self._result_label = QLabel()
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        # Fallback manual instructions (for failure case, initially hidden)
        self._fallback_widget = QWidget()
        fallback_layout = QVBoxLayout(self._fallback_widget)
        fallback_layout.setContentsMargins(0, 0, 0, 0)
        fallback_layout.setSpacing(8)

        fallback_header = QLabel("<b>You can download the model manually:</b>")
        fallback_layout.addWidget(fallback_header)

        fallback_btn_layout = QHBoxLayout()

        self._copy_link_btn = QPushButton("Copy Download Link")
        self._copy_link_btn.clicked.connect(self._copy_download_link)
        fallback_btn_layout.addWidget(self._copy_link_btn)

        fallback_open_folder_btn = QPushButton("Open Models Folder")
        fallback_open_folder_btn.clicked.connect(self._open_models_folder)
        fallback_btn_layout.addWidget(fallback_open_folder_btn)

        fallback_layout.addLayout(fallback_btn_layout)
        self._fallback_widget.hide()
        layout.addWidget(self._fallback_widget)

        # Close button (for failure case, initially hidden)
        close_btn_layout = QHBoxLayout()
        close_btn_layout.addStretch()

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.reject)
        self._close_btn.hide()
        close_btn_layout.addWidget(self._close_btn)

        layout.addLayout(close_btn_layout)

        return widget

    def _on_checkbox_changed(self) -> None:
        """Enable Download button when checkbox is checked."""
        self._download_btn.setEnabled(self._accept_checkbox.isChecked())

    def _start_download(self) -> None:
        """Transition to Phase 2: Start the download task."""
        # Hide confirmation, show download progress
        self._confirmation_widget.hide()
        self._result_widget.hide()
        self._download_widget.show()

        card = self._card
        models_dir = self._models_dir

        def worker(token, handle):
            from caliscope.trackers.model_download import download_and_extract_model

            return download_and_extract_model(
                card,
                models_dir,
                progress_callback=lambda downloaded, total: handle.report_progress(
                    int(downloaded / total * 100) if total > 0 else 0,
                    f"Downloading... {downloaded // (1024 * 1024)} MB"
                    + (f" / {total // (1024 * 1024)} MB" if total > 0 else ""),
                ),
                cancellation_check=lambda: token.is_cancelled,
            )

        self._task_handle = self._task_manager.submit(worker, name=f"download_{card.name}", auto_start=False)

        # Connect signals with QueuedConnection (signals come from worker thread)
        self._task_handle.completed.connect(
            self._on_download_complete,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.failed.connect(
            self._on_download_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.cancelled.connect(
            self._on_download_cancelled,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_handle.progress_updated.connect(
            self._on_progress,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_manager.start_task(self._task_handle.task_id)

    def _cancel_download(self) -> None:
        """Cancel the running download task."""
        if self._task_handle is not None:
            self._task_handle.cancel()

    def _on_progress(self, percent: int, message: str) -> None:
        """Update progress bar and status label."""
        # When total bytes unknown, percent will be 0 and message won't have " / "
        if percent == 0 and " / " not in message:
            # Indeterminate progress (no total size known)
            self._progress_bar.setRange(0, 0)
        else:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(percent)

        self._status_label.setText(message)

    def _on_download_complete(self, result) -> None:
        """Transition to Phase 3a: Success."""
        # Hide download progress, show success message
        self._download_widget.hide()
        self._result_widget.show()
        self._fallback_widget.hide()
        self._close_btn.hide()

        self._result_label.setText("<b>Download complete!</b>")

        # Auto-close after 1.5 seconds
        QTimer.singleShot(1500, self.accept)

    def _on_download_failed(self, exc_type: str, message: str) -> None:
        """Transition to Phase 3b: Failure with fallback instructions."""
        # Hide download progress, show error and fallback
        self._download_widget.hide()
        self._result_widget.show()
        self._fallback_widget.show()
        self._close_btn.show()

        self._result_label.setText(f"<b>Download failed:</b> {exc_type}<br>{message}")

    def _on_download_cancelled(self) -> None:
        """Transition to Phase 3c: Cancelled - revert to Phase 1."""
        # Hide download progress, show confirmation again
        self._download_widget.hide()
        self._result_widget.hide()
        self._confirmation_widget.show()

        # Reset progress UI
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._status_label.setText("Downloading...")

        self._task_handle = None

    def _copy_download_link(self) -> None:
        """Copy the model download URL to clipboard."""
        if self._card.source_url:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(self._card.source_url)

    def _open_models_folder(self) -> None:
        """Open the models directory in the system file manager."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._models_dir)))
