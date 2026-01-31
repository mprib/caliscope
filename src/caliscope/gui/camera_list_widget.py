"""Camera list widget with calibration status indicators.

Displays a list of cameras with visual feedback on their calibration state:
- Green dot + RMSE for calibrated cameras
- Hollow circle for uncalibrated cameras

Uses both icons and color for accessibility (not color-alone).
"""

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from caliscope.cameras.camera_array import CameraArray

logger = logging.getLogger(__name__)


class CameraListWidget(QListWidget):
    """Sidebar list showing cameras with calibration status indicators.

    Emits camera_selected(port) when the user selects a different camera.
    """

    camera_selected = Signal(int)  # port number

    def __init__(self, camera_array: CameraArray):
        super().__init__()
        self._camera_array = camera_array
        self._port_to_row: dict[int, int] = {}

        # Style for comfortable row height and strong selection highlight
        self.setStyleSheet("""
            QListWidget::item {
                padding: 8px 12px;
                min-height: 24px;
            }
            QListWidget::item:selected {
                background-color: #3a5f8a;
            }
        """)

        self._populate_list()
        self.currentRowChanged.connect(self._on_row_changed)

    def _populate_list(self) -> None:
        """Build list items from camera array."""
        self.clear()
        self._port_to_row.clear()

        for row, (port, camera) in enumerate(sorted(self._camera_array.cameras.items())):
            self._port_to_row[port] = row
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, port)

            if camera.matrix is not None and camera.distortions is not None:
                # Calibrated: filled circle + green text + optional RMSE
                if camera.error is not None:
                    text = f"\u25cf Port {port} \u2014 {camera.error:.2f}px"
                else:
                    text = f"\u25cf Port {port}"
                item.setForeground(QBrush(QColor("#4CAF50")))  # Material green
            else:
                # Not calibrated: hollow circle + red text
                text = f"\u25cb Port {port}"
                item.setForeground(QBrush(QColor("#F44336")))  # Material red

            item.setText(text)
            self.addItem(item)

    def _on_row_changed(self, row: int) -> None:
        """Handle selection change and emit camera_selected signal."""
        if row < 0:
            return  # No selection

        item = self.item(row)
        if item is not None:
            port = item.data(Qt.ItemDataRole.UserRole)
            logger.info(f"Camera selected: port {port}")
            self.camera_selected.emit(port)

    def refresh(self, camera_array: CameraArray) -> None:
        """Refresh the list with updated camera array data.

        Preserves current selection if possible. Blocks signals during restore
        to prevent re-triggering camera_selected (which would destroy the
        current presenter).
        """
        current_item = self.currentItem()
        current_port = current_item.data(Qt.ItemDataRole.UserRole) if current_item else None

        self._camera_array = camera_array
        self._populate_list()

        # Restore selection with signals blocked - we're just updating the
        # visual state, not selecting a different camera
        if current_port is not None and current_port in self._port_to_row:
            self.blockSignals(True)
            try:
                self.setCurrentRow(self._port_to_row[current_port])
            finally:
                self.blockSignals(False)

    def select_port(self, port: int) -> None:
        """Programmatically select a camera by port number."""
        if port in self._port_to_row:
            self.setCurrentRow(self._port_to_row[port])
