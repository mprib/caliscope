"""Compact sparkline chart for relative distance error visualization."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from caliscope.core.scale_accuracy import VolumetricScaleReport
from caliscope.gui.theme import Colors


class DistanceSparkline(QWidget):
    """Compact sparkline showing per-frame relative distance error across the capture volume.

    Displays:
    - Filled area chart of per-frame relative RMSE (% of object size)
    - Vertical cursor at current slider position
    - Y-max label in top-left corner
    - Placeholder text when no data available

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

        self.setFixedHeight(40)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)  # Enable hover tooltips

    def set_data(self, report: VolumetricScaleReport, valid_sync_indices: np.ndarray) -> None:
        """Update sparkline with new report data, mapped into the slider's position domain.

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

    def clear(self) -> None:
        """Clear sparkline and show placeholder."""
        self._report = None
        self._valid_sync_indices = np.array([], dtype=np.int64)
        self._position_data = {}
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Draw the sparkline chart."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background
            painter.fillRect(self.rect(), QColor(Colors.SURFACE_DARK))

            # Show placeholder if no data
            if not self._position_data:
                self._draw_placeholder(painter)
                return

            # Draw filled area chart
            self._draw_chart(painter)

            # Draw cursor
            self._draw_cursor(painter)

            # Draw y-max label
            self._draw_y_max_label(painter)
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

    def _max_position(self) -> int:
        """Highest valid slider position (domain upper bound)."""
        return max(len(self._valid_sync_indices) - 1, 0)

    def _draw_chart(self, painter: QPainter) -> None:
        """Draw the filled area chart with gaps for positions missing data."""
        if not self._position_data:
            return

        max_position = self._max_position()
        if max_position == 0:
            self._draw_single_point(painter)
            return

        max_val = max(self._position_data.values())
        if max_val == 0:
            max_val = 1.0

        width = self.width()
        height = self.height()
        margin = 4

        def pos_to_x(position: int) -> float:
            return margin + position / max_position * (width - 2 * margin)

        def val_to_y(value: float) -> float:
            return height - margin - (value / max_val) * (height - 2 * margin)

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

            baseline_y = height - margin
            for position in reversed(segment):
                points.append((pos_to_x(position), baseline_y))

            polygon = QPolygonF([QPointF(x, y) for x, y in points])
            painter.drawPolygon(polygon)

    def _draw_single_point(self, painter: QPainter) -> None:
        """Draw a single dot when the slider domain has only one position."""
        if not self._position_data:
            return

        value = next(iter(self._position_data.values()))
        max_val = max(value, 0.1)

        width = self.width()
        height = self.height()
        margin = 4

        x = width / 2
        y = height - margin - (value / max_val) * (height - 2 * margin)

        primary_color = QColor(Colors.PRIMARY)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(primary_color)
        painter.drawEllipse(int(x - 3), int(y - 3), 6, 6)

    def _draw_cursor(self, painter: QPainter) -> None:
        """Draw vertical dashed line at current slider position."""
        max_position = self._max_position()

        if not (0 <= self._cursor_position <= max_position):
            return

        if max_position == 0:
            x = self.width() / 2
        else:
            margin = 4
            width = self.width()
            x = margin + self._cursor_position / max_position * (width - 2 * margin)

        pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), 0, int(x), self.height())

    def _draw_y_max_label(self, painter: QPainter) -> None:
        """Draw y-max label in top-left corner."""
        if not self._position_data:
            return

        y_max = max(self._position_data.values())
        text = f"{y_max:.1f}%"

        painter.setPen(QColor(Colors.TEXT_MUTED))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        margin = 4
        painter.drawText(margin, margin, self.width() - 2 * margin, 15, Qt.AlignmentFlag.AlignLeft, text)

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

        max_position = self._max_position()
        if max_position == 0:
            return next(iter(self._position_data.keys()))

        margin = 4
        width = self.width()

        normalized_x = (x - margin) / (width - 2 * margin)
        normalized_x = max(0.0, min(1.0, normalized_x))  # Clamp to [0, 1]
        position_float = normalized_x * max_position

        # Find nearest position with actual data
        available_positions = sorted(self._position_data.keys())
        nearest = min(available_positions, key=lambda p: abs(p - position_float))

        return nearest
