"""Quality panel widget displaying calibration metrics."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from caliscope.core.scale_accuracy import ScaleAccuracyData
from caliscope.gui.presenters.extrinsic_calibration_presenter import QualityPanelData


class QualityPanel(QWidget):
    """Display calibration quality metrics with three groups in a horizontal row.

    Layout:
    [Reprojection Error] [Scale Accuracy] [Per-Camera Table]

    Each group maintains its own detailed internal format.

    Usage:
        panel = QualityPanel()
        panel.set_reprojection_data(quality_data)
        panel.set_scale_accuracy(scale_data)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the widget layout with three horizontal groups."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Group 1: Reprojection Error (global metrics)
        reproj_group = QGroupBox("Reprojection Error")
        reproj_layout = QVBoxLayout(reproj_group)
        reproj_layout.setContentsMargins(8, 4, 8, 4)
        reproj_layout.setSpacing(2)

        self._rmse_label = QLabel("RMSE: --")
        self._rmse_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        reproj_layout.addWidget(self._rmse_label)

        self._obs_label = QLabel("Observations: --")
        reproj_layout.addWidget(self._obs_label)

        self._points_label = QLabel("3D Points: --")
        reproj_layout.addWidget(self._points_label)

        self._converged_label = QLabel("Converged: --")
        reproj_layout.addWidget(self._converged_label)

        reproj_layout.addStretch()
        layout.addWidget(reproj_group)

        # Group 2: Scale Accuracy
        scale_group = QGroupBox("Scale Accuracy")
        scale_layout = QVBoxLayout(scale_group)
        scale_layout.setContentsMargins(8, 4, 8, 4)
        scale_layout.setSpacing(2)

        self._scale_rmse_label = QLabel("Distance RMSE: --")
        self._scale_rmse_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        scale_layout.addWidget(self._scale_rmse_label)

        self._scale_relative_label = QLabel("Relative Error: --")
        scale_layout.addWidget(self._scale_relative_label)

        self._scale_ref_label = QLabel("Reference Frame: --")
        scale_layout.addWidget(self._scale_ref_label)

        self._scale_detail_label = QLabel("")
        self._scale_detail_label.setStyleSheet("color: #888888; font-style: italic;")
        scale_layout.addWidget(self._scale_detail_label)

        scale_layout.addStretch()
        layout.addWidget(scale_group)

        # Group 3: Per-Camera Table
        self._camera_group = QGroupBox("Per-Camera")
        camera_layout = QVBoxLayout(self._camera_group)
        camera_layout.setContentsMargins(4, 4, 4, 4)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Camera", "Observations", "RMSE (px)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        # Allow table to show more rows, stretch to fill
        self._table.setMinimumHeight(80)

        camera_layout.addWidget(self._table)
        layout.addWidget(self._camera_group, stretch=1)  # Table gets extra space

    def set_reprojection_data(self, data: QualityPanelData) -> None:
        """Update reprojection error section with new data.

        Args:
            data: Quality metrics from the presenter
        """
        self._rmse_label.setText(f"RMSE: {data.overall_rmse:.3f} px")
        self._obs_label.setText(f"Observations: {data.n_observations:,}")
        self._points_label.setText(f"3D Points: {data.n_world_points:,}")

        status = "Yes" if data.converged else "No"
        self._converged_label.setText(f"Converged: {status} ({data.iterations} iter)")

        # Update per-camera table
        self._table.setRowCount(len(data.camera_rows))
        for row, (port, n_obs, rmse) in enumerate(data.camera_rows):
            port_item = QTableWidgetItem(f"Port {port}")
            port_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            obs_item = QTableWidgetItem(f"{n_obs:,}")
            obs_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            rmse_item = QTableWidgetItem(f"{rmse:.3f}")
            rmse_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            self._table.setItem(row, 0, port_item)
            self._table.setItem(row, 1, obs_item)
            self._table.setItem(row, 2, rmse_item)

    def set_scale_accuracy(self, data: ScaleAccuracyData | None) -> None:
        """Update scale accuracy section with new data.

        Args:
            data: Scale accuracy metrics, or None to show placeholder
        """
        if data is None or data.n_corners_detected == 0:
            self._scale_rmse_label.setText("Distance RMSE: --")
            self._scale_relative_label.setText("Relative Error: --")
            self._scale_ref_label.setText("Reference Frame: --")
            self._scale_detail_label.setText("(set origin to compute)")
            return

        self._scale_rmse_label.setText(f"Distance RMSE: {data.distance_rmse_mm:.2f} mm")
        self._scale_relative_label.setText(f"Relative Error: {data.relative_error_percent:.2f}%")
        self._scale_ref_label.setText(f"Reference Frame: {data.reference_sync_index}")
        self._scale_detail_label.setText(f"({data.n_corners_detected} corners matched)")

    def clear(self) -> None:
        """Reset all displays to placeholder values."""
        self._rmse_label.setText("RMSE: --")
        self._obs_label.setText("Observations: --")
        self._points_label.setText("3D Points: --")
        self._converged_label.setText("Converged: --")

        self._scale_rmse_label.setText("Distance RMSE: --")
        self._scale_relative_label.setText("Relative Error: --")
        self._scale_ref_label.setText("Reference Frame: --")
        self._scale_detail_label.setText("")

        self._table.setRowCount(0)
