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
    """Display calibration quality metrics.

    Shows three sections:
    1. Reprojection Error: Overall RMSE, observation counts, convergence status
    2. Scale Accuracy: Distance RMSE vs ground truth (when reference frame set)
    3. Per-Camera Table: Observations and RMSE per camera

    Usage:
        panel = QualityPanel()
        panel.set_reprojection_data(quality_data)
        panel.set_scale_accuracy(scale_data)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Top row: Reprojection Error + Scale Accuracy side by side
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Section 1: Reprojection Error
        self._reprojection_group = QGroupBox("Reprojection Error")
        repro_layout = QVBoxLayout(self._reprojection_group)
        repro_layout.setSpacing(4)

        self._rmse_label = QLabel("Overall RMSE: --")
        self._obs_label = QLabel("Observations: --")
        self._points_label = QLabel("3D Points: --")
        self._converged_label = QLabel("Converged: --")

        repro_layout.addWidget(self._rmse_label)
        repro_layout.addWidget(self._obs_label)
        repro_layout.addWidget(self._points_label)
        repro_layout.addWidget(self._converged_label)

        top_row.addWidget(self._reprojection_group)

        # Section 2: Scale Accuracy
        self._scale_group = QGroupBox("Scale Accuracy")
        scale_layout = QVBoxLayout(self._scale_group)
        scale_layout.setSpacing(4)

        self._scale_ref_label = QLabel("Reference: --")
        self._scale_rmse_label = QLabel("Distance RMSE: --")
        self._scale_relative_label = QLabel("Relative Error: --")
        self._scale_detail_label = QLabel("")
        self._scale_detail_label.setStyleSheet("color: #888888;")

        scale_layout.addWidget(self._scale_ref_label)
        scale_layout.addWidget(self._scale_rmse_label)
        scale_layout.addWidget(self._scale_relative_label)
        scale_layout.addWidget(self._scale_detail_label)

        top_row.addWidget(self._scale_group)
        layout.addLayout(top_row)

        # Section 3: Per-Camera Table
        self._camera_group = QGroupBox("Per-Camera Metrics")
        camera_layout = QVBoxLayout(self._camera_group)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Camera", "Observations", "RMSE (px)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        # Set minimum height to show ~4 rows comfortably, no max so it can expand
        self._table.setMinimumHeight(120)

        camera_layout.addWidget(self._table)
        layout.addWidget(self._camera_group)

        layout.addStretch()

    def set_reprojection_data(self, data: QualityPanelData) -> None:
        """Update reprojection error section with new data.

        Args:
            data: Quality metrics from the presenter
        """
        self._rmse_label.setText(f"Overall RMSE: {data.overall_rmse:.2f} px")
        self._obs_label.setText(f"Observations: {data.n_observations:,}")
        self._points_label.setText(f"3D Points: {data.n_world_points:,}")

        status = "Yes" if data.converged else "No"
        self._converged_label.setText(f"Converged: {status} ({data.iterations} iterations)")

        # Update per-camera table
        self._table.setRowCount(len(data.camera_rows))
        for row, (port, n_obs, rmse) in enumerate(data.camera_rows):
            port_item = QTableWidgetItem(f"Port {port}")
            port_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            obs_item = QTableWidgetItem(f"{n_obs:,}")
            obs_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            rmse_item = QTableWidgetItem(f"{rmse:.2f}")
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
            self._scale_ref_label.setText("Reference: not set")
            self._scale_rmse_label.setText("Distance RMSE: --")
            self._scale_relative_label.setText("Relative Error: --")
            self._scale_detail_label.setText("Set origin frame to compute scale accuracy")
            return

        self._scale_ref_label.setText(f"Reference: frame {data.reference_sync_index}")
        self._scale_rmse_label.setText(f"Distance RMSE: {data.distance_rmse_mm:.2f} mm")
        self._scale_relative_label.setText(f"Relative Error: {data.relative_error_percent:.2f}%")
        self._scale_detail_label.setText(f"({data.n_corners_detected} corners, {data.n_distance_pairs} distance pairs)")

    def clear(self) -> None:
        """Reset all displays to placeholder values."""
        self._rmse_label.setText("Overall RMSE: --")
        self._obs_label.setText("Observations: --")
        self._points_label.setText("3D Points: --")
        self._converged_label.setText("Converged: --")

        self._scale_ref_label.setText("Reference: --")
        self._scale_rmse_label.setText("Distance RMSE: --")
        self._scale_relative_label.setText("Relative Error: --")
        self._scale_detail_label.setText("")

        self._table.setRowCount(0)
