"""Quality panel widget displaying calibration metrics."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from caliscope.core.scale_accuracy import VolumetricScaleReport
from caliscope.gui.presenters.extrinsic_calibration_presenter import QualityPanelData


class QualityPanel(QWidget):
    """Display calibration quality metrics with three sections in a horizontal row.

    Layout:
    [Reprojection Error] [Scale Accuracy] [Per-Camera Table]

    Uses typography and spacing for visual separation - no heavy borders.
    Each section is evenly distributed horizontally.

    Usage:
        panel = QualityPanel()
        panel.set_reprojection_data(quality_data)
        panel.set_volumetric_accuracy(report)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the widget layout with three horizontal sections."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(24)  # Generous spacing between sections

        # Section 1: Reprojection Error (global metrics)
        reproj_section = QWidget()
        reproj_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        reproj_layout = QVBoxLayout(reproj_section)
        reproj_layout.setContentsMargins(0, 0, 0, 0)
        reproj_layout.setSpacing(4)

        reproj_header = QLabel("Reprojection Error")
        reproj_header.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        reproj_layout.addWidget(reproj_header)

        self._rmse_label = QLabel("RMSE: --")
        self._rmse_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        reproj_layout.addWidget(self._rmse_label)

        self._obs_label = QLabel("Observations: --")
        self._obs_label.setStyleSheet("color: #ccc;")
        reproj_layout.addWidget(self._obs_label)

        self._points_label = QLabel("3D Points: --")
        self._points_label.setStyleSheet("color: #ccc;")
        reproj_layout.addWidget(self._points_label)

        self._converged_label = QLabel("Converged: --")
        self._converged_label.setStyleSheet("color: #ccc;")
        reproj_layout.addWidget(self._converged_label)

        reproj_layout.addStretch()
        layout.addWidget(reproj_section, stretch=1)

        # Section 2: Scale Accuracy
        scale_section = QWidget()
        scale_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        scale_layout = QVBoxLayout(scale_section)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(4)

        scale_header = QLabel("Scale Accuracy")
        scale_header.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        scale_layout.addWidget(scale_header)

        # Primary metric: pooled RMSE (14px bold)
        self._scale_pooled_rmse_label = QLabel("Pooled RMSE: --")
        self._scale_pooled_rmse_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        scale_layout.addWidget(self._scale_pooled_rmse_label)

        # Secondary metrics: median + worst (one line)
        self._scale_median_worst_label = QLabel("Median: -- | Worst: --")
        self._scale_median_worst_label.setStyleSheet("color: #ccc;")
        scale_layout.addWidget(self._scale_median_worst_label)

        # Tertiary: bias + frames (muted/italic)
        self._scale_bias_frames_label = QLabel("Bias: -- | -- frames")
        self._scale_bias_frames_label.setStyleSheet("color: #888; font-style: italic;")
        scale_layout.addWidget(self._scale_bias_frames_label)

        scale_layout.addStretch()
        layout.addWidget(scale_section, stretch=1)

        # Section 3: Per-Camera Table
        camera_section = QWidget()
        camera_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        camera_layout = QVBoxLayout(camera_section)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(4)

        camera_header = QLabel("Per-Camera")
        camera_header.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        camera_layout.addWidget(camera_header)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Camera", "Observations", "RMSE (px)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setMinimumHeight(80)
        self._table.setMaximumHeight(120)
        # Subtle styling - no heavy borders
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                border: none;
                gridline-color: #333;
            }
            QHeaderView::section {
                background-color: #252525;
                border: none;
                border-bottom: 1px solid #333;
                padding: 4px;
            }
        """)

        camera_layout.addWidget(self._table)
        layout.addWidget(camera_section, stretch=2)  # Table gets more space

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

    def set_volumetric_accuracy(self, report: VolumetricScaleReport | None) -> None:
        """Update scale accuracy section with volumetric report.

        Args:
            report: Volumetric scale accuracy report, or None to show placeholder
        """
        if report is None or report.n_frames_sampled == 0:
            self._scale_pooled_rmse_label.setText("Pooled RMSE: --")
            self._scale_median_worst_label.setText("Median: -- | Worst: --")
            self._scale_bias_frames_label.setText("(set origin to compute)")
            return

        # Primary: pooled RMSE
        self._scale_pooled_rmse_label.setText(f"Pooled RMSE: {report.pooled_rmse_mm:.2f} mm")

        # Secondary: median + worst
        self._scale_median_worst_label.setText(
            f"Median: {report.median_rmse_mm:.2f} mm | Worst: {report.max_rmse_mm:.1f} mm"
        )

        # Tertiary: bias + frames (sign prefix on bias)
        self._scale_bias_frames_label.setText(
            f"Bias: {report.mean_signed_error_mm:+.2f} mm | {report.n_frames_sampled} frames"
        )

    def clear(self) -> None:
        """Reset all displays to placeholder values."""
        self._rmse_label.setText("RMSE: --")
        self._obs_label.setText("Observations: --")
        self._points_label.setText("3D Points: --")
        self._converged_label.setText("Converged: --")

        self._scale_pooled_rmse_label.setText("Pooled RMSE: --")
        self._scale_median_worst_label.setText("Median: -- | Worst: --")
        self._scale_bias_frames_label.setText("(set origin to compute)")

        self._table.setRowCount(0)
