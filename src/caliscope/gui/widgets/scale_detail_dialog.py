"""Expanded detail dialog for volumetric scale accuracy visualization."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QDialog, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget

from caliscope.core.scale_accuracy import VolumetricScaleReport
from caliscope.gui.theme import Colors


class ScaleDetailChartWidget(QWidget):
    """QPainter-based chart for scale accuracy with axes, gridlines, and tooltips.

    Displays:
    - Filled area chart of per-frame RMSE
    - Axis labels (X = "Frame", Y = "RMSE (mm)")
    - Horizontal and vertical gridlines
    - Vertical cursor at current frame
    - Hover tooltip: "Frame: N | RMSE: X.Xmm | Cameras: N"

    Interaction:
    - Hover: tooltip showing frame number, RMSE, and camera count
    - Click: emits frame_clicked signal for click-to-seek
    """

    frame_clicked = Signal(int)  # sync_index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._report: VolumetricScaleReport | None = None
        self._cursor_sync_index: int = 0
        self._frame_data: dict[int, tuple[float, int]] = {}  # sync_index -> (rmse, n_cameras)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)  # Enable hover tooltips

    def set_data(self, report: VolumetricScaleReport) -> None:
        """Update chart with new report data.

        Args:
            report: Volumetric scale accuracy report with per-frame errors
        """
        self._report = report

        # Pre-build frame_data lookup for efficient painting
        self._frame_data = {
            fe.sync_index: (fe.distance_rmse_mm, fe.n_cameras_contributing) for fe in report.frame_errors
        }

        self.update()

    def set_cursor(self, sync_index: int) -> None:
        """Update cursor position.

        Args:
            sync_index: Current frame sync_index
        """
        self._cursor_sync_index = sync_index
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Draw the chart with axes, gridlines, and data."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background
            painter.fillRect(self.rect(), QColor(Colors.SURFACE_DARK))

            # Show placeholder if no data
            if self._report is None or self._report.n_frames_sampled == 0:
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
        if self._report is None or self._report.n_frames_sampled == 0:
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
        painter.drawText(-50, 0, 100, 20, Qt.AlignmentFlag.AlignCenter, "RMSE (mm)")
        painter.restore()

        # Y-axis tick labels
        max_rmse = self._report.max_rmse_mm
        if max_rmse > 0:
            font.setPointSize(9)
            painter.setFont(font)

            # Draw 3-5 tick labels
            n_ticks = 5
            for i in range(n_ticks + 1):
                value = (max_rmse / n_ticks) * i
                y = y_offset + height - (i / n_ticks) * height
                text = f"{value:.1f}"
                painter.drawText(5, int(y - 8), 40, 16, Qt.AlignmentFlag.AlignRight, text)

    def _draw_gridlines(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Draw horizontal and vertical gridlines."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return

        pen = QPen(QColor(Colors.BORDER_SUBTLE), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)

        # Horizontal gridlines (5 lines)
        n_h_lines = 5
        for i in range(1, n_h_lines):
            y = y_offset + (i / n_h_lines) * height
            painter.drawLine(x_offset, int(y), x_offset + width, int(y))

        # Vertical gridlines (based on frame range)
        min_sync = self._report.min_sync_index
        max_sync = self._report.max_sync_index
        frame_range = max_sync - min_sync

        if frame_range > 0:
            # Aim for ~5-10 vertical lines
            n_v_lines = min(10, max(5, frame_range // 20))
            for i in range(1, n_v_lines):
                x = x_offset + (i / n_v_lines) * width
                painter.drawLine(int(x), y_offset, int(x), y_offset + height)

    def _draw_chart(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Draw the filled area chart with gaps for missing frames."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return

        # Calculate scaling
        min_sync = self._report.min_sync_index
        max_sync = self._report.max_sync_index
        frame_range = max_sync - min_sync
        if frame_range == 0:
            # Single frame - draw a dot
            self._draw_single_point(painter, x_offset, y_offset, width, height)
            return

        max_rmse = self._report.max_rmse_mm
        if max_rmse == 0:
            max_rmse = 1.0  # Avoid division by zero

        # Helper: sync_index -> x pixel
        def sync_to_x(sync_idx: int) -> float:
            return x_offset + (sync_idx - min_sync) / frame_range * width

        # Helper: rmse -> y pixel (inverted: 0 at bottom)
        def rmse_to_y(rmse: float) -> float:
            return y_offset + height - (rmse / max_rmse) * height

        # Group contiguous frames into segments (gaps = missing sync_indices)
        segments: list[list[int]] = []
        sorted_indices = sorted(self._frame_data.keys())

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
                rmse, _ = self._frame_data[sync_idx]
                points.append((sync_to_x(sync_idx), rmse_to_y(rmse)))

            # Baseline edge (right to left)
            baseline_y = y_offset + height
            for sync_idx in reversed(segment):
                points.append((sync_to_x(sync_idx), baseline_y))

            polygon = QPolygonF([QPointF(x, y) for x, y in points])
            painter.drawPolygon(polygon)

    def _draw_single_point(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
        """Draw a single dot when only one frame is available."""
        if not self._frame_data or self._report is None:
            return

        sync_idx = list(self._frame_data.keys())[0]
        rmse, _ = self._frame_data[sync_idx]

        x = x_offset + width / 2  # Center horizontally
        y = y_offset + height - (rmse / max(self._report.max_rmse_mm, 1.0)) * height

        primary_color = QColor(Colors.PRIMARY)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(primary_color)
        painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)

    def _draw_cursor(self, painter: QPainter, x_offset: int, y_offset: int, width: int, height: int) -> None:
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
            x = x_offset + width / 2
        else:
            x = x_offset + (self._cursor_sync_index - min_sync) / frame_range * width

        # Draw white dashed line
        pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), y_offset, int(x), y_offset + height)

    def mouseMoveEvent(self, event) -> None:
        """Show tooltip on hover."""
        if self._report is None or self._report.n_frames_sampled == 0:
            return

        # Find nearest frame to cursor position
        x = event.pos().x()
        nearest_frame = self._sync_index_at_x(x)

        if nearest_frame is not None and nearest_frame in self._frame_data:
            rmse, camera_count = self._frame_data[nearest_frame]
            tooltip_text = f"Frame: {nearest_frame} | RMSE: {rmse:.1f}mm | Cameras: {camera_count}"
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

        margin_left = 50
        margin_right = 10
        chart_width = self.width() - margin_left - margin_right

        min_sync = self._report.min_sync_index
        max_sync = self._report.max_sync_index
        frame_range = max_sync - min_sync

        if frame_range == 0:
            return min_sync  # Single frame

        # Map x to sync_index
        normalized_x = (x - margin_left) / chart_width
        normalized_x = max(0.0, min(1.0, normalized_x))  # Clamp to [0, 1]
        sync_float = min_sync + normalized_x * frame_range

        # Find nearest actual frame
        available_frames = sorted(self._frame_data.keys())
        nearest = min(available_frames, key=lambda s: abs(s - sync_float))

        return nearest


class ScaleDetailDialog(QDialog):
    """Modeless dialog showing expanded scale accuracy chart and aggregate statistics.

    Layout (top to bottom):
    1. Chart area (~300px, expanding): Large QPainter chart with axes and gridlines
    2. Stats summary (~60px, fixed): Pooled RMSE, Median RMSE, Worst RMSE, Bias, Frames Sampled

    Signals:
    - frame_clicked(int): Emitted when user clicks chart (for click-to-seek)
    """

    frame_clicked = Signal(int)  # sync_index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Scale Accuracy Detail")
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

    def set_data(self, report: VolumetricScaleReport) -> None:
        """Update dialog with new report data.

        Args:
            report: Volumetric scale accuracy report
        """
        self._chart.set_data(report)

        # Build stats summary text
        if report.n_frames_sampled == 0:
            stats_text = "No data available"
        else:
            stats_text = (
                f"<b>Pooled RMSE:</b> {report.pooled_rmse_mm:.2f} mm  |  "
                f"<b>Median RMSE:</b> {report.median_rmse_mm:.2f} mm  |  "
                f"<b>Worst RMSE:</b> {report.max_rmse_mm:.2f} mm  |  "
                f"<b>Bias:</b> {report.mean_signed_error_mm:+.2f} mm  |  "
                f"<b>Frames Sampled:</b> {report.n_frames_sampled}"
            )

        self._stats_label.setText(stats_text)

    def set_cursor(self, sync_index: int) -> None:
        """Update cursor position on chart.

        Args:
            sync_index: Current frame sync_index
        """
        self._chart.set_cursor(sync_index)
