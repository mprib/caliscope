"""Expanded detail dialog for volumetric distance accuracy visualization."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QDialog, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget

from caliscope.core.scale_accuracy import VolumetricScaleReport
from caliscope.gui.theme import Colors


class ScaleDetailChartWidget(QWidget):
    """QPainter-based chart for relative distance error with axes, gridlines, and tooltips.

    Displays:
    - Filled area chart of per-frame relative RMSE (% of object size)
    - Axis labels (X = "Frame", Y = "Relative Error (%)")
    - Horizontal and vertical gridlines
    - Vertical cursor at current slider position
    - Hover tooltip: "frame N · X.XX% · Y.Y mm"

    The x-axis is the slider's position domain (0..n-1), not sync_index, so the
    cursor stays aligned with the frame slider even when sync indices are sparse.

    Interaction:
    - Hover: tooltip showing frame number, relative error, and mm error
    - Click: emits frame_clicked signal for click-to-seek
    """

    frame_clicked = Signal(int)  # slider position

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._report: VolumetricScaleReport | None = None
        self._valid_sync_indices: np.ndarray = np.array([], dtype=np.int64)
        self._position_data: dict[int, float] = {}  # position -> relative RMSE %

        self._cursor_position: int = 0

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)  # Enable hover tooltips

    def set_data(self, report: VolumetricScaleReport, valid_sync_indices: np.ndarray) -> None:
        """Update chart with new report data, mapped into the slider's position domain.

        Args:
            report: Volumetric scale accuracy report with per-frame errors
            valid_sync_indices: Sync indices in slider order; array index is the slider position
        """
        self._report = report
        self._valid_sync_indices = valid_sync_indices

        per_frame_pct = report.per_frame_relative_rmse_pct
        self._position_data = {}
        for position, sync_index in enumerate(valid_sync_indices):
            sync_index_int = int(sync_index)
            if sync_index_int in per_frame_pct:
                self._position_data[position] = per_frame_pct[sync_index_int]

        self.update()

    def set_cursor(self, position: int) -> None:
        """Update cursor position.

        Args:
            position: Current slider position (0-based index into valid_sync_indices)
        """
        self._cursor_position = position
        self.update()

    def _max_position(self) -> int:
        """Highest valid slider position (domain upper bound)."""
        return max(len(self._valid_sync_indices) - 1, 0)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Draw the chart with axes, gridlines, and data."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background
            painter.fillRect(self.rect(), QColor(Colors.SURFACE_DARK))

            # Show placeholder if no data
            if not self._position_data:
                self._draw_placeholder(painter)
                return

            # Calculate chart area (reserve space for axes)
            margin_left = 50
            margin_bottom = 40
            margin_top = 10
            margin_right = 10

            chart_width = self.width() - margin_left - margin_right
            chart_height = self.height() - margin_top - margin_bottom

            if chart_width <= 0 or chart_height <= 0:
                return

            # Draw gridlines first (behind data)
            self._draw_gridlines(painter, margin_left, margin_top, chart_width, chart_height)

            # Draw filled area chart
            self._draw_chart(painter, margin_left, margin_top, chart_width, chart_height)

            # Draw cursor
            self._draw_cursor(painter, margin_left, margin_top, chart_width, chart_height)

            # Draw axes last (on top)
            self._draw_axes(painter, margin_left, margin_top, chart_width, chart_height)
        finally:
            painter.end()

    def _draw_placeholder(self, painter: QPainter) -> None:
        """Draw placeholder text when no data available."""
        painter.setPen(QColor(Colors.TEXT_MUTED))
        font = QFont()
        font.setItalic(True)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Set origin to compute scale accuracy",
        )

    def _draw_axes(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Draw axis labels and tick marks."""
        if not self._position_data:
            return

        painter.setPen(QColor(Colors.TEXT_SECONDARY))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)

        # X-axis label
        painter.drawText(
            x_offset,
            y_offset + height + 5,
            width,
            30,
            Qt.AlignmentFlag.AlignCenter,
            "Frame",
        )

        # Y-axis label (rotated)
        painter.save()
        painter.translate(10, y_offset + height // 2)
        painter.rotate(-90)
        painter.drawText(-60, 0, 120, 20, Qt.AlignmentFlag.AlignCenter, "Relative Error (%)")
        painter.restore()

        # Y-axis tick labels
        max_val = max(self._position_data.values())
        if max_val > 0:
            font.setPointSize(9)
            painter.setFont(font)

            # Draw 3-5 tick labels
            n_ticks = 5
            for i in range(n_ticks + 1):
                value = (max_val / n_ticks) * i
                y = y_offset + height - (i / n_ticks) * height
                text = f"{value:.1f}"
                painter.drawText(5, int(y - 8), 40, 16, Qt.AlignmentFlag.AlignRight, text)

    def _draw_gridlines(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Draw horizontal and vertical gridlines."""
        if not self._position_data:
            return

        pen = QPen(QColor(Colors.BORDER_SUBTLE), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)

        # Horizontal gridlines (5 lines)
        n_h_lines = 5
        for i in range(1, n_h_lines):
            y = y_offset + (i / n_h_lines) * height
            painter.drawLine(x_offset, int(y), x_offset + width, int(y))

        # Vertical gridlines (based on slider position range)
        max_position = self._max_position()

        if max_position > 0:
            # Aim for ~5-10 vertical lines
            n_v_lines = min(10, max(5, max_position // 20))
            for i in range(1, n_v_lines):
                x = x_offset + (i / n_v_lines) * width
                painter.drawLine(int(x), y_offset, int(x), y_offset + height)

    def _draw_chart(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Draw the filled area chart with gaps for positions missing data."""
        if not self._position_data:
            return

        max_position = self._max_position()
        if max_position == 0:
            # Single position - draw a dot
            self._draw_single_point(painter, x_offset, y_offset, width, height)
            return

        max_val = max(self._position_data.values())
        if max_val == 0:
            max_val = 1.0

        # Helper: position -> x pixel
        def pos_to_x(position: int) -> float:
            return x_offset + position / max_position * width

        # Helper: value -> y pixel (inverted: 0 at bottom)
        def val_to_y(value: float) -> float:
            return y_offset + height - (value / max_val) * height

        # Group contiguous positions into segments (gaps = positions with no data)
        segments: list[list[int]] = []
        sorted_positions = sorted(self._position_data.keys())

        current_segment: list[int] = []
        for i, position in enumerate(sorted_positions):
            if i == 0:
                current_segment = [position]
            elif position == sorted_positions[i - 1] + 1:
                current_segment.append(position)
            else:
                segments.append(current_segment)
                current_segment = [position]

        if current_segment:
            segments.append(current_segment)

        # Draw each segment as a filled polygon
        primary_color = QColor(Colors.PRIMARY)
        fill_color = QColor(primary_color)
        fill_color.setAlpha(64)  # 25% opacity

        pen = QPen(primary_color, 1.5)
        painter.setPen(pen)
        painter.setBrush(fill_color)

        for segment in segments:
            points: list[tuple[float, float]] = []

            for position in segment:
                points.append((pos_to_x(position), val_to_y(self._position_data[position])))

            baseline_y = y_offset + height
            for position in reversed(segment):
                points.append((pos_to_x(position), baseline_y))

            polygon = QPolygonF([QPointF(x, y) for x, y in points])
            painter.drawPolygon(polygon)

    def _draw_single_point(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Draw a single dot when the slider domain has only one position."""
        if not self._position_data:
            return

        value = next(iter(self._position_data.values()))
        max_val = max(value, 0.1)

        x = x_offset + width / 2  # Center horizontally
        y = y_offset + height - (value / max_val) * height

        primary_color = QColor(Colors.PRIMARY)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(primary_color)
        painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)

    def _draw_cursor(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Draw vertical dashed line at current slider position."""
        max_position = self._max_position()

        if not (0 <= self._cursor_position <= max_position):
            return

        if max_position == 0:
            x = x_offset + width / 2
        else:
            x = x_offset + self._cursor_position / max_position * width

        # Draw white dashed line
        pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), y_offset, int(x), y_offset + height)

    def mouseMoveEvent(self, event) -> None:
        """Show tooltip on hover."""
        if self._report is None or not self._position_data:
            return

        position = self._position_at_x(event.pos().x())
        if position is None or position not in self._position_data:
            return

        pct = self._position_data[position]
        sync_index = int(self._valid_sync_indices[position])
        mm = self._report.per_frame_rmse_mm.get(sync_index, 0.0)

        tooltip_text = f"frame {sync_index} · {pct:.2f}% · {mm:.1f} mm"
        QToolTip.showText(event.globalPos(), tooltip_text, self)

    def mousePressEvent(self, event) -> None:
        """Emit frame_clicked signal on click."""
        if not self._position_data:
            return

        position = self._position_at_x(event.pos().x())
        if position is not None:
            self.frame_clicked.emit(position)

    def _position_at_x(self, x: float) -> int | None:
        """Find nearest slider position with data at given x pixel."""
        if not self._position_data:
            return None

        margin_left = 50
        margin_right = 10
        chart_width = self.width() - margin_left - margin_right

        max_position = self._max_position()
        if max_position == 0:
            return next(iter(self._position_data.keys()))

        # Map x to position
        normalized_x = (x - margin_left) / chart_width
        normalized_x = max(0.0, min(1.0, normalized_x))  # Clamp to [0, 1]
        position_float = normalized_x * max_position

        # Find nearest position with actual data
        available_positions = sorted(self._position_data.keys())
        nearest = min(available_positions, key=lambda p: abs(p - position_float))

        return nearest


class ScaleDetailDialog(QDialog):
    """Modeless dialog showing expanded distance error chart and aggregate statistics.

    Layout (top to bottom):
    1. Chart area (~300px, expanding): Large QPainter chart with axes and gridlines
    2. Stats summary (~60px, fixed): Pooled relative RMSE, pooled RMSE, signed mean error, frames sampled

    Signals:
    - frame_clicked(int): Emitted when user clicks chart (for click-to-seek), as slider position
    """

    frame_clicked = Signal(int)  # slider position

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Distance Error Detail")
        self.setModal(False)  # Modeless - allow interaction with main window
        self.setMinimumSize(500, 400)
        self.resize(500, 400)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the dialog layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Chart widget
        self._chart = ScaleDetailChartWidget()
        self._chart.setMinimumHeight(300)
        self._chart.frame_clicked.connect(self.frame_clicked)
        layout.addWidget(self._chart, stretch=1)

        # Stats summary
        self._stats_label = QLabel()
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self._stats_label)

    def set_data(self, report: VolumetricScaleReport, valid_sync_indices: np.ndarray) -> None:
        """Update dialog with new report data.

        Args:
            report: Volumetric scale accuracy report
            valid_sync_indices: Sync indices in slider order; array index is the slider position
        """
        self._chart.set_data(report, valid_sync_indices)

        # Build stats summary text
        if report.n_frames_sampled == 0:
            stats_text = "No data available"
        else:
            stats_text = (
                f"<b>Pooled Relative RMSE:</b> {report.pooled_relative_rmse_pct:.2f}%  |  "
                f"<b>Pooled RMSE:</b> {report.pooled_rmse_mm:.2f} mm  |  "
                f"<b>Signed Mean Error:</b> {report.mean_signed_error_mm:+.2f} mm  |  "
                f"<b>Frames Sampled:</b> {report.n_frames_sampled}"
            )

        self._stats_label.setText(stats_text)

    def set_cursor(self, position: int) -> None:
        """Update cursor position on chart.

        Args:
            position: Current slider position (0-based index into valid_sync_indices)
        """
        self._chart.set_cursor(position)
