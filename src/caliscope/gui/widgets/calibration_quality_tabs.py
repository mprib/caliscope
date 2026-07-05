"""Calibration quality report widget.

Displays reprojection error, distance-based scale accuracy, and per-camera /
per-marker breakdowns in a tabbed layout embedded in the extrinsic
calibration view.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from caliscope.gui.presenters.extrinsic_calibration_presenter import CalibrationQualityData
from caliscope.gui.theme import Colors, Styles

CAMERA_COLUMN_TOOLTIPS = {
    1: "Number of 2D point observations for this camera across all frames",
    2: "Root mean square reprojection error in pixels — lower is better",
    3: "Focal length in pixels after calibration",
    4: "Radial distortion coefficients",
    5: "Radial distortion coefficients",
    6: (
        "How the camera's intrinsics were determined: 'provided (locked)' = used as-is, "
        "'provided (refined)' = re-estimated during calibration, "
        "'estimated' = no prior intrinsics available"
    ),
}

MARKER_COLUMN_TOOLTIPS = {
    1: "Whether this marker was treated as fixed (static) or mobile during calibration",
    2: "Relative distance error as percentage of the marker's largest diagonal",
    3: "Absolute distance error in millimeters",
    4: "Number of point-to-point distances measured",
}


class CalibrationQualityTabs(QWidget):
    """Tabbed calibration quality report: Summary, Cameras, and (optionally) Markers.

    Populated from a single `CalibrationQualityData` snapshot via `set_data()`.
    The Markers tab is shown only when marker rows are present.
    """

    view_lens_model_requested = Signal(int)  # cam_id
    view_coverage_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tab_widget = QTabWidget()
        layout.addWidget(self._tab_widget)

        self._summary_tab = self._build_summary_tab()
        self._tab_widget.addTab(self._summary_tab, "Summary")

        self._cameras_tab = self._build_cameras_tab()
        self._tab_widget.addTab(self._cameras_tab, "Cameras")

        # Markers tab is added/removed dynamically in set_data() based on data availability.
        self._markers_tab = self._build_markers_tab()

    # -------------------------------------------------------------------------
    # Summary tab
    # -------------------------------------------------------------------------

    def _build_summary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        headline_row = QHBoxLayout()

        self._rmse_label = QLabel("—")
        self._rmse_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        headline_row.addWidget(self._rmse_label)

        desc = QLabel("reprojection RMSE")
        desc.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px; padding-top: 4px;")
        headline_row.addWidget(desc)

        headline_row.addSpacing(16)

        self._meta_label = QLabel()
        self._meta_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px; padding-top: 4px;")
        headline_row.addWidget(self._meta_label)

        headline_row.addStretch()
        layout.addLayout(headline_row)

        self._distance_error_label = QLabel()
        self._distance_error_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        layout.addWidget(self._distance_error_label)
        self._distance_error_label.hide()

        self._breakdown_label = QLabel()
        self._breakdown_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        layout.addWidget(self._breakdown_label)
        self._breakdown_label.hide()

        self._warnings_band = QWidget()
        self._warnings_band.setStyleSheet(
            f"background-color: rgba(255, 160, 0, 0.08); border-left: 3px solid {Colors.WARNING}; padding: 8px 12px;"
        )
        self._warnings_layout = QVBoxLayout(self._warnings_band)
        self._warnings_layout.setContentsMargins(12, 4, 4, 4)
        self._warnings_layout.setSpacing(2)
        layout.addWidget(self._warnings_band)
        self._warnings_band.hide()

        layout.addStretch()

        self._footer_label = QLabel()
        self._footer_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._footer_label)

        return tab

    def _update_summary_tab(self, data: CalibrationQualityData) -> None:
        self._rmse_label.setText(f"{data.overall_rmse_px:.2f} px")

        converge_text = "converged" if data.converged else "did not converge"
        self._meta_label.setText(
            f"{data.n_observations:,} observations · {converge_text} ({data.iterations} iterations)"
        )

        if data.distance_error_pct is not None:
            mm_suffix = f" ({data.distance_error_mm:.1f} mm RMSE)" if data.distance_error_mm is not None else ""
            self._distance_error_label.setText(
                f"Distance error: {data.distance_error_pct:.2f}% of object size{mm_suffix}"
            )
            self._distance_error_label.show()
        else:
            self._distance_error_label.hide()

        if data.moving_error_pct is not None and data.static_error_pct is not None:
            self._breakdown_label.setText(f"moving {data.moving_error_pct:.2f}% · static {data.static_error_pct:.2f}%")
            self._breakdown_label.show()
        else:
            self._breakdown_label.hide()

        self._set_warnings(data.warnings)

        footer_text = f"{data.n_world_points:,} world points"
        if data.filter_summary:
            footer_text += f" · {data.filter_summary}"
        self._footer_label.setText(footer_text)

    def _set_warnings(self, warnings: list[str]) -> None:
        while self._warnings_layout.count():
            item = self._warnings_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        if not warnings:
            self._warnings_band.hide()
            return

        for warning in warnings:
            label = QLabel(warning)
            label.setStyleSheet(f"color: {Colors.WARNING}; background: transparent;")
            label.setWordWrap(True)
            self._warnings_layout.addWidget(label)
        self._warnings_band.show()

    # -------------------------------------------------------------------------
    # Cameras tab
    # -------------------------------------------------------------------------

    def _build_cameras_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.addStretch()
        self._view_coverage_btn = QPushButton("View Coverage…")
        self._view_coverage_btn.setStyleSheet(Styles.GHOST_BUTTON)
        self._view_coverage_btn.clicked.connect(self.view_coverage_requested)
        header_row.addWidget(self._view_coverage_btn)
        layout.addLayout(header_row)

        self._camera_table = QTableWidget()
        columns = ["Camera", "Obs", "RMSE (px)", "f (px)", "k1", "k2", "Source", ""]
        self._camera_table.setColumnCount(len(columns))
        self._camera_table.setHorizontalHeaderLabels(columns)
        self._camera_table.verticalHeader().setVisible(False)
        self._camera_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._camera_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._camera_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._camera_table.setMaximumHeight(250)

        header = self._camera_table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(columns) - 1):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(columns) - 1, QHeaderView.ResizeMode.Fixed)
        self._camera_table.setColumnWidth(len(columns) - 1, 70)

        for col, tip in CAMERA_COLUMN_TOOLTIPS.items():
            item = self._camera_table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)

        self._apply_table_style(self._camera_table)
        layout.addWidget(self._camera_table)
        layout.addStretch()

        return tab

    def _update_cameras_tab(self, data: CalibrationQualityData) -> None:
        rows = data.camera_rows
        self._camera_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            cell_values = [
                (f"Cam {row.cam_id}", Qt.AlignmentFlag.AlignCenter),
                (f"{row.n_observations:,}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (f"{row.rmse_px:.1f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (f"{row.f_px:.0f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (f"{row.k1:.3f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (f"{row.k2:.3f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (row.source, Qt.AlignmentFlag.AlignCenter),
            ]
            for col, (text, align) in enumerate(cell_values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                self._camera_table.setItem(row_idx, col, item)

            lens_btn = QPushButton("Lens…")
            lens_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
            cam_id = row.cam_id
            lens_btn.clicked.connect(lambda checked=False, cid=cam_id: self.view_lens_model_requested.emit(cid))
            self._camera_table.setCellWidget(row_idx, len(cell_values), lens_btn)

    # -------------------------------------------------------------------------
    # Markers tab
    # -------------------------------------------------------------------------

    def _build_markers_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._marker_table = QTableWidget()
        columns = ["Marker", "Type", "Distance error (% of size)", "RMSE (mm)", "Pairs"]
        self._marker_table.setColumnCount(len(columns))
        self._marker_table.setHorizontalHeaderLabels(columns)
        self._marker_table.verticalHeader().setVisible(False)
        self._marker_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._marker_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._marker_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._marker_table.setMaximumHeight(250)

        header = self._marker_table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

        for col, tip in MARKER_COLUMN_TOOLTIPS.items():
            item = self._marker_table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)

        self._apply_table_style(self._marker_table)
        layout.addWidget(self._marker_table)
        layout.addStretch()

        return tab

    def _update_markers_tab(self, data: CalibrationQualityData) -> None:
        rows = data.marker_rows

        self._marker_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            cell_values = [
                (str(row.object_id), Qt.AlignmentFlag.AlignCenter),
                ("static" if row.is_static else "moving", Qt.AlignmentFlag.AlignCenter),
                (f"{row.relative_rmse_pct:.1f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (f"{row.rmse_mm:.0f}", Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                (str(row.n_pairs), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            ]
            for col, (text, align) in enumerate(cell_values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                self._marker_table.setItem(row_idx, col, item)

        if rows:
            if self._tab_widget.indexOf(self._markers_tab) == -1:
                self._tab_widget.addTab(self._markers_tab, "Markers")
        else:
            idx = self._tab_widget.indexOf(self._markers_tab)
            if idx != -1:
                self._tab_widget.removeTab(idx)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_data(self, data: CalibrationQualityData) -> None:
        """Populate all tabs from a calibration quality snapshot."""
        self._update_summary_tab(data)
        self._update_cameras_tab(data)
        self._update_markers_tab(data)

    def clear(self) -> None:
        """Reset all displays to placeholder values."""
        self._rmse_label.setText("—")
        self._meta_label.setText("")
        self._distance_error_label.hide()
        self._breakdown_label.hide()
        self._set_warnings([])
        self._footer_label.setText("")

        self._camera_table.setRowCount(0)
        self._marker_table.setRowCount(0)
        if self._markers_tab_index is not None:
            self._tab_widget.removeTab(self._markers_tab_index)
            self._markers_tab_index = None

    def set_disabled_with_last_values(self, disabled: bool) -> None:
        """Disable interaction while retaining the currently displayed values.

        Used during filter re-optimization: the last-good numbers stay visible
        (greyed out via Qt's disabled styling) rather than being cleared.
        """
        self._tab_widget.setEnabled(not disabled)

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
            }}
        """)
