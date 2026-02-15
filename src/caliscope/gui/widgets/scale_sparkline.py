"""Compact sparkline chart for volumetric scale accuracy visualization."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from caliscope.core.scale_accuracy import VolumetricScaleReport
from caliscope.gui.theme import Colors


class ScaleSparkline(QWidget):
    """Compact sparkline showing per-frame distance RMSE across the capture volume.

    Displays:
    - Filled area chart of per-frame RMSE
    - Vertical cursor at current frame
    - Y-max label in top-left corner
    - Placeholder text when no data available

    Interaction:
    - Hover: tooltip showing frame number and RMSE
    - Click: emits frame_clicked signal for click-to-seek
    """

    frame_clicked = Signal(int)  # sync_index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._report: VolumetricScaleReport | None = None
        self._cursor_sync_index: int = 0

        self.setFixedHeight(40)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)  # Enable hover tooltips

    def set_data(self, report: VolumetricScaleReport) -> None:
        """Update sparkline with new report data.

        Args:
            report: Volumetric scale accuracy report with per-frame errors
        """
        self._report = report
        self.update()

    def set_cursor(self, sync_index: int) -> None:
        """Update cursor position.

        Args:
            sync_index: Current frame sync_index
        """
        self._cursor_sync_index = sync_index
        self.update()

    def clear(self) -> None:
        """Clear sparkline and show placeholder."""
        self._report = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Draw the sparkline chart."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background
            painter.fillRect(self.rect(), QColor(Colors.SURFACE_DARK))

            # Show placeholder if no data
            if self._report is None or self._report.n_frames_sampled == 0:
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

    def _draw_chart(self, painter: QPainter) -> None:
        """Draw the filled area chart with gaps for missing frames."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return

        # Build sync_index -> rmse lookup
        frame_data = {fe.sync_index: fe.distance_rmse_mm for fe in self._report.frame_errors}

        # Calculate scaling
        min_sync = self._report.min_sync_index
        max_sync = self._report.max_sync_index
        frame_range = max_sync - min_sync
        if frame_range == 0:
            # Single frame - draw a dot
            self._draw_single_point(painter, frame_data)
            return

        max_rmse = self._report.max_rmse_mm
        if max_rmse == 0:
            max_rmse = 1.0  # Avoid division by zero

        width = self.width()
        height = self.height()
        margin = 4

        # Helper: sync_index -> x pixel
        def sync_to_x(sync_idx: int) -> float:
            return margin + (sync_idx - min_sync) / frame_range * (width - 2 * margin)

        # Helper: rmse -> y pixel (inverted: 0 at bottom)
        def rmse_to_y(rmse: float) -> float:
            return height - margin - (rmse / max_rmse) * (height - 2 * margin)

        # Group contiguous frames into segments (gaps = missing sync_indices)
        segments: list[list[int]] = []
        sorted_indices = sorted(frame_data.keys())

        current_segment: list[int] = []
        for i, sync_idx in enumerate(sorted_indices):
            if i == 0:
                current_segment = [sync_idx]
            elif sync_idx == sorted_indices[i - 1] + 1:
                # Contiguous
                current_segment.append(sync_idx)
            else:
                # Gap detected
                segments.append(current_segment)
                current_segment = [sync_idx]

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
            # Build polygon points: line from left to right, then baseline back
            points = []

            # Top edge (line chart)
            for sync_idx in segment:
                rmse = frame_data[sync_idx]
                points.append((sync_to_x(sync_idx), rmse_to_y(rmse)))

            # Baseline edge (right to left)
            baseline_y = height - margin
            for sync_idx in reversed(segment):
                points.append((sync_to_x(sync_idx), baseline_y))

            polygon = QPolygonF([QPointF(x, y) for x, y in points])
            painter.drawPolygon(polygon)

    def _draw_single_point(self, painter: QPainter, frame_data: dict[int, float]) -> None:
        """Draw a single dot when only one frame is available."""
        if not frame_data or self._report is None:
            return

        sync_idx = list(frame_data.keys())[0]
        rmse = frame_data[sync_idx]

        width = self.width()
        height = self.height()
        margin = 4

        x = width / 2  # Center horizontally
        y = height - margin - (rmse / max(self._report.max_rmse_mm, 1.0)) * (height - 2 * margin)

        primary_color = QColor(Colors.PRIMARY)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(primary_color)
        painter.drawEllipse(int(x - 3), int(y - 3), 6, 6)

    def _draw_cursor(self, painter: QPainter) -> None:
        """Draw vertical dashed line at current sync_index."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return

        min_sync = self._report.min_sync_index
        max_sync = self._report.max_sync_index
        frame_range = max_sync - min_sync

        # Check if cursor is in range
        if not (min_sync <= self._cursor_sync_index <= max_sync):
            return

        if frame_range == 0:
            # Single frame - cursor is at center
            x = self.width() / 2
        else:
            margin = 4
            width = self.width()
            x = margin + (self._cursor_sync_index - min_sync) / frame_range * (width - 2 * margin)

        # Draw white dashed line
        pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), 0, int(x), self.height())

    def _draw_y_max_label(self, painter: QPainter) -> None:
        """Draw y-max label in top-left corner."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return

        max_rmse = self._report.max_rmse_mm
        text = f"{max_rmse:.1f}mm"

        painter.setPen(QColor(Colors.TEXT_MUTED))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        margin = 4
        painter.drawText(margin, margin, self.width() - 2 * margin, 15, Qt.AlignmentFlag.AlignLeft, text)

    def mouseMoveEvent(self, event) -> None:
        """Show tooltip on hover."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return

        # Find nearest frame to cursor position
        x = event.pos().x()
        nearest_frame = self._sync_index_at_x(x)

        if nearest_frame is not None:
            # Find RMSE for this frame
            frame_rmse = None
            for fe in self._report.frame_errors:
                if fe.sync_index == nearest_frame:
                    frame_rmse = fe.distance_rmse_mm
                    break

            if frame_rmse is not None:
                tooltip_text = f"Frame: {nearest_frame} | RMSE: {frame_rmse:.1f}mm"
                QToolTip.showText(event.globalPos(), tooltip_text, self)

    def mousePressEvent(self, event) -> None:
        """Emit frame_clicked signal on click."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return

        x = event.pos().x()
        nearest_frame = self._sync_index_at_x(x)

        if nearest_frame is not None:
            self.frame_clicked.emit(nearest_frame)

    def _sync_index_at_x(self, x: float) -> int | None:
        """Find sync_index at given x position (nearest frame)."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return None

        min_sync = self._report.min_sync_index
        max_sync = self._report.max_sync_index
        frame_range = max_sync - min_sync

        if frame_range == 0:
            return min_sync  # Single frame

        margin = 4
        width = self.width()

        # Map x to sync_index
        normalized_x = (x - margin) / (width - 2 * margin)
        normalized_x = max(0.0, min(1.0, normalized_x))  # Clamp to [0, 1]
        sync_float = min_sync + normalized_x * frame_range

        # Find nearest actual frame
        available_frames = sorted([fe.sync_index for fe in self._report.frame_errors])
        nearest = min(available_frames, key=lambda s: abs(s - sync_float))

        return nearest
