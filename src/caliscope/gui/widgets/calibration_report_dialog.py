"""Calibration quality report dialog.

Shows calibration metrics: reprojection error (headline), warnings,
per-camera parameters with lens visualization, and rigidity summary.
Receives a flat display model from the presenter.
"""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    CalibrationReportData,
)
from caliscope.gui.theme import Colors, Typography

HELP_TEXT = {
    "f": "Focal length in pixels. Determines field of view.",
    "k1": "First radial distortion coefficient. Negative = barrel, positive = pincushion.",
    "k2": "Second radial distortion coefficient. Usually smaller than k1.",
    "source": "Whether intrinsics were provided by the user or estimated from scratch by the optimizer.",
    "depth_ratio": (
        "Ratio of scene depth to camera-subject distance. Higher values give better focal length recovery."
    ),
    "rel_error": (
        "Distance error as a percentage of expected distance. "
        "Under 1% is excellent, 1-3% is typical, above 5% is suspect."
    ),
}

WARNING_CAM_PATTERN = re.compile(r"Camera (\d+):")


class CalibrationReportDialog(QDialog):
    """Non-modal dialog showing detailed calibration quality metrics."""

    view_lens_model_requested = Signal(int)  # cam_id

    def __init__(self, data: CalibrationReportData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data = data
        self.setWindowTitle("Calibration Report")
        self.setModal(False)
        self.setMinimumWidth(650)

        self._warned_cam_ids: set[int] = set()
        for w in data.warnings:
            m = WARNING_CAM_PATTERN.search(w)
            if m:
                self._warned_cam_ids.add(int(m.group(1)))

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        self._add_summary_header(layout)
        self._add_warning_band(layout)
        self._add_camera_table(layout)
        self._add_marker_rigidity(layout)
        layout.addStretch()

    def _add_summary_header(self, layout: QVBoxLayout) -> None:
        """Reprojection error as the headline metric."""
        row = QHBoxLayout()

        rmse_label = QLabel(f"{self._data.overall_rmse:.3f} px")
        rmse_label.setStyleSheet("font-weight: bold; font-size: 18px; color: #fff;")
        row.addWidget(rmse_label)

        desc = QLabel("reprojection RMSE")
        desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 12px; padding-top: 4px;")
        row.addWidget(desc)

        row.addSpacing(24)

        obs_label = QLabel(f"{self._data.n_observations:,} observations")
        obs_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding-top: 4px;")
        row.addWidget(obs_label)

        converge_text = "converged" if self._data.converged else "did not converge"
        conv_label = QLabel(f"· {converge_text}")
        conv_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; padding-top: 4px;")
        row.addWidget(conv_label)

        row.addStretch()
        layout.addLayout(row)

    def _add_warning_band(self, layout: QVBoxLayout) -> None:
        if not self._data.warnings:
            return

        band = QWidget()
        band.setStyleSheet(
            f"background-color: rgba(255, 160, 0, 0.08); border-left: 3px solid {Colors.WARNING}; padding: 8px 12px;"
        )
        band_layout = QVBoxLayout(band)
        band_layout.setContentsMargins(12, 4, 4, 4)
        band_layout.setSpacing(2)

        for warning in self._data.warnings:
            lbl = QLabel(warning)
            lbl.setStyleSheet(f"color: {Colors.WARNING}; background: transparent;")
            lbl.setWordWrap(True)
            band_layout.addWidget(lbl)

        layout.addWidget(band)

    def _add_camera_table(self, layout: QVBoxLayout) -> None:
        rows = self._data.intrinsic_rows
        if not rows:
            return

        depth_by_cam = {r.cam_id: r.depth_ratio for r in self._data.camera_detail_rows}

        layout.addWidget(self._section_header("Camera Parameters"))

        table = QTableWidget()
        cols = ["Camera", "f (px)", "k1", "k2", "Source", "Depth Ratio", ""]
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)

        header = table.horizontalHeader()
        assert header is not None
        for i in range(len(cols) - 1):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(len(cols) - 1, QHeaderView.ResizeMode.ResizeToContents)

        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._apply_table_style(table)

        col_help = {
            1: HELP_TEXT["f"],
            2: HELP_TEXT["k1"],
            3: HELP_TEXT["k2"],
            4: HELP_TEXT["source"],
            5: HELP_TEXT["depth_ratio"],
        }
        for col_idx, tip in col_help.items():
            item = table.horizontalHeaderItem(col_idx)
            if item:
                item.setToolTip(tip)

        amber_brush = QBrush(QColor(Colors.WARNING))
        table.setRowCount(len(rows))
        for row_idx, r in enumerate(rows):
            is_warned = r.cam_id in self._warned_cam_ids
            depth = depth_by_cam.get(r.cam_id)

            items_data = [
                (f"Cam {r.cam_id}", Qt.AlignmentFlag.AlignCenter),
                (f"{r.f:.1f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (f"{r.k1:.4f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (f"{r.k2:.4f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (r.source, Qt.AlignmentFlag.AlignCenter),
                (
                    f"{depth:.2f}×" if depth is not None else "—",
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                ),
            ]

            for col, (text, align) in enumerate(items_data):
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                if is_warned:
                    item.setForeground(amber_brush)
                table.setItem(row_idx, col, item)

            btn = QPushButton("Lens ▸")
            btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
            cam_id = r.cam_id
            btn.clicked.connect(lambda checked, cid=cam_id: self.view_lens_model_requested.emit(cid))
            table.setCellWidget(row_idx, len(cols) - 1, btn)

        table.setMaximumHeight(36 + 28 * len(rows))
        layout.addWidget(table)

    def _add_marker_rigidity(self, layout: QVBoxLayout) -> None:
        rows = self._data.marker_rigidity_rows
        has_drops = len(self._data.dropped_static_markers) > 0

        if not rows and self._data.rigidity_relative_pct is None:
            return

        header_row = QHBoxLayout()
        header_row.addWidget(self._section_header("Marker Rigidity"))

        if self._data.rigidity_relative_pct is not None:
            summary = f"{self._data.rigidity_relative_pct:.1f}% relative RMSE"
            if self._data.rigidity_rmse_mm is not None:
                summary += f" · {self._data.rigidity_rmse_mm:.1f} mm"
            summary += f" · {self._data.n_constraints} constraints"
            summary_label = QLabel(summary)
            summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
            summary_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            header_row.addWidget(summary_label, stretch=1)

        layout.addLayout(header_row)

        info = QLabel("Relative RMSE = distance error / expected distance, across all constraints and frames")
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(info)

        if not rows:
            return

        self._rigidity_table_widget = QWidget()
        table_layout = QVBoxLayout(self._rigidity_table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Marker", "Type", "Rel. Error (%)"])
        header = table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._apply_table_style(table)

        rel_header = table.horizontalHeaderItem(2)
        if rel_header:
            rel_header.setToolTip(HELP_TEXT["rel_error"])

        table.setRowCount(len(rows))
        for row_idx, r in enumerate(rows):
            id_item = QTableWidgetItem(str(r.object_id))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row_idx, 0, id_item)

            type_item = QTableWidgetItem("Static" if r.is_static else "Moving")
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row_idx, 1, type_item)

            rel_item = QTableWidgetItem(f"{r.relative_error_pct:.1f}%")
            rel_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_idx, 2, rel_item)

        table.setMaximumHeight(36 + 28 * len(rows))
        table_layout.addWidget(table)

        self._rigidity_toggle = QPushButton("▸ Show per-marker detail")
        self._rigidity_toggle.setStyleSheet(
            f"text-align: left; color: {Colors.PRIMARY}; background: transparent; "
            f"border: none; padding: 2px 0; font-size: 11px;"
        )
        self._rigidity_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rigidity_toggle.clicked.connect(self._toggle_rigidity_table)
        layout.addWidget(self._rigidity_toggle)

        layout.addWidget(self._rigidity_table_widget)

        if has_drops:
            self._rigidity_table_widget.show()
            self._rigidity_toggle.setText("▾ Hide per-marker detail")
        else:
            self._rigidity_table_widget.hide()

    def _toggle_rigidity_table(self) -> None:
        visible = self._rigidity_table_widget.isVisible()
        self._rigidity_table_widget.setVisible(not visible)
        if visible:
            self._rigidity_toggle.setText("▸ Show per-marker detail")
        else:
            self._rigidity_toggle.setText("▾ Hide per-marker detail")

    def _section_header(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(Typography.SECTION_HEADER)
        return label

    def _apply_table_style(self, table: QTableWidget) -> None:
        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Colors.SURFACE_DARK};
                border: none;
                gridline-color: {Colors.BORDER_SUBTLE};
            }}
            QHeaderView::section {{
                background-color: #252525;
                border: none;
                border-bottom: 1px solid {Colors.BORDER_SUBTLE};
                padding: 4px;
                color: {Colors.PRIMARY};
                text-decoration: underline;
            }}
        """)
