"""Compact sparkline chart for per-frame rigidity constraint error."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from caliscope.gui.theme import Colors


class RigiditySparkline(QWidget):
    """Sparkline showing per-frame relative constraint error (%).

    Same visual pattern as ScaleSparkline: filled area chart, vertical
    cursor synced to the frame slider, click-to-seek.
    """

    frame_clicked = Signal(int)  # sync_index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._frame_data: dict[int, float] = {}  # sync_index -> relative RMSE %
        self._cursor_sync_index: int = 0

        self.setFixedHeight(40)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def set_data(self, per_frame_pct: dict[int, float]) -> None:
        self._frame_data = per_frame_pct
        self.update()

    def set_cursor(self, sync_index: int) -> None:
        self._cursor_sync_index = sync_index
        self.update()

    def clear(self) -> None:
        self._frame_data = {}
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(Colors.SURFACE_DARK))

            if not self._frame_data:
                self._draw_placeholder(painter)
                return

            self._draw_chart(painter)
            self._draw_cursor(painter)
            self._draw_y_max_label(painter)
        finally:
            painter.end()

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor(Colors.TEXT_MUTED))
        font = QFont()
        font.setItalic(True)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "No rigidity constraints",
        )

    def _draw_chart(self, painter: QPainter) -> None:
        sorted_indices = sorted(self._frame_data.keys())
        if len(sorted_indices) < 2:
            self._draw_single_point(painter)
            return

        min_sync = sorted_indices[0]
        max_sync = sorted_indices[-1]
        frame_range = max_sync - min_sync
        max_val = max(self._frame_data.values())
        if max_val == 0:
            max_val = 1.0

        width = self.width()
        height = self.height()
        margin = 4

        def sync_to_x(si: int) -> float:
            return margin + (si - min_sync) / frame_range * (width - 2 * margin)

        def val_to_y(v: float) -> float:
            return height - margin - (v / max_val) * (height - 2 * margin)

        # Segment contiguous frames
        segments: list[list[int]] = []
        current: list[int] = []
        for i, si in enumerate(sorted_indices):
            if i == 0:
                current = [si]
            elif si == sorted_indices[i - 1] + 1:
                current.append(si)
            else:
                segments.append(current)
                current = [si]
        if current:
            segments.append(current)

        accent = QColor("#e6a817")  # amber-gold, distinct from scale sparkline's blue
        fill = QColor(accent)
        fill.setAlpha(64)

        painter.setPen(QPen(accent, 1.5))
        painter.setBrush(fill)

        baseline_y = height - margin
        for seg in segments:
            points: list[tuple[float, float]] = []
            for si in seg:
                points.append((sync_to_x(si), val_to_y(self._frame_data[si])))
            for si in reversed(seg):
                points.append((sync_to_x(si), baseline_y))
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))

    def _draw_single_point(self, painter: QPainter) -> None:
        if not self._frame_data:
            return
        accent = QColor("#e6a817")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        x = self.width() / 2
        val = list(self._frame_data.values())[0]
        max_val = max(val, 0.1)
        y = self.height() - 4 - (val / max_val) * (self.height() - 8)
        painter.drawEllipse(int(x - 3), int(y - 3), 6, 6)

    def _draw_cursor(self, painter: QPainter) -> None:
        if not self._frame_data:
            return
        sorted_indices = sorted(self._frame_data.keys())
        min_sync = sorted_indices[0]
        max_sync = sorted_indices[-1]
        frame_range = max_sync - min_sync

        if not (min_sync <= self._cursor_sync_index <= max_sync):
            return

        margin = 4
        if frame_range == 0:
            x = self.width() / 2
        else:
            x = margin + (self._cursor_sync_index - min_sync) / frame_range * (self.width() - 2 * margin)

        pen = QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), 0, int(x), self.height())

    def _draw_y_max_label(self, painter: QPainter) -> None:
        if not self._frame_data:
            return
        max_val = max(self._frame_data.values())
        painter.setPen(QColor(Colors.TEXT_MUTED))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(4, 4, self.width() - 8, 15, Qt.AlignmentFlag.AlignLeft, f"{max_val:.1f}%")

    def mouseMoveEvent(self, event) -> None:
        if not self._frame_data:
            return
        x = event.pos().x()
        nearest = self._sync_index_at_x(x)
        if nearest is not None and nearest in self._frame_data:
            val = self._frame_data[nearest]
            QToolTip.showText(event.globalPos(), f"Frame: {nearest} | Rel. RMSE: {val:.2f}%", self)

    def mousePressEvent(self, event) -> None:
        if not self._frame_data:
            return
        nearest = self._sync_index_at_x(event.pos().x())
        if nearest is not None:
            self.frame_clicked.emit(nearest)

    def _sync_index_at_x(self, x: float) -> int | None:
        if not self._frame_data:
            return None
        sorted_indices = sorted(self._frame_data.keys())
        min_sync = sorted_indices[0]
        max_sync = sorted_indices[-1]
        frame_range = max_sync - min_sync

        if frame_range == 0:
            return min_sync

        margin = 4
        normalized = (x - margin) / (self.width() - 2 * margin)
        normalized = max(0.0, min(1.0, normalized))
        sync_float = min_sync + normalized * frame_range
        return min(sorted_indices, key=lambda s: abs(s - sync_float))
