"""Per-camera 2D observation view for the Synthetic Calibration Explorer.

One small panel per camera shows the image points that camera observes at the
currently selected frame, in image coordinates, coloured by object_id. For a
thick two-sided board this makes the culling visible: each camera sees a single
flat face (one colour) even though the 3D storyboard shows the two faces moving
together as a thin box.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    from caliscope.synthetic.synthetic_scene import SyntheticScene

# Distinct hue per face so a glance shows which side a camera is looking at.
_FACE_COLORS: dict[int, QColor] = {
    0: QColor(80, 170, 255),  # front face - blue
    1: QColor(255, 150, 60),  # back face - orange
}
_FACE_LABELS: dict[int, str] = {0: "front", 1: "back"}
_FALLBACK_COLOR = QColor(180, 180, 180)

_PANEL_PADDING = 6
_TITLE_HEIGHT = 16
_LEGEND_HEIGHT = 22
_POINT_RADIUS = 2.5


class PerCameraObservationsView(QWidget):
    """Grid of per-camera 2D scatter panels for the selected frame.

    Holds a reference to the immutable scene and re-derives the observed points
    whenever the scene or frame changes. Painting only reads the prepared
    per-camera point lists.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._scene: SyntheticScene | None = None
        self._frame: int = 0
        self._cam_ids: list[int] = []
        self._cam_sizes: dict[int, tuple[int, int]] = {}
        # cam_id -> list of (img_x, img_y, object_id)
        self._points_by_cam: dict[int, list[tuple[float, float, int]]] = {}

        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_scene(self, scene: SyntheticScene) -> None:
        """Bind a new scene and rebuild the current frame's observations."""
        self._scene = scene
        self._cam_ids = sorted(scene.camera_array.cameras.keys())
        self._cam_sizes = {cam_id: scene.camera_array.cameras[cam_id].size for cam_id in self._cam_ids}
        self._frame = 0
        self._rebuild()

    def set_frame(self, frame: int) -> None:
        """Show the observations at the given sync index."""
        self._frame = frame
        self._rebuild()

    def _rebuild(self) -> None:
        self._points_by_cam = {cam_id: [] for cam_id in self._cam_ids}

        if self._scene is not None:
            df = self._scene.image_points_noisy.df
            frame_df = df[df["sync_index"] == self._frame]
            for cam_id in self._cam_ids:
                cam_df = frame_df[frame_df["cam_id"] == cam_id]
                self._points_by_cam[cam_id] = [
                    (float(x), float(y), int(obj))
                    for x, y, obj in zip(cam_df["img_loc_x"], cam_df["img_loc_y"], cam_df["object_id"])
                ]

        self.update()

    def _object_ids_present(self) -> list[int]:
        seen: set[int] = set()
        for points in self._points_by_cam.values():
            for _, _, obj in points:
                seen.add(obj)
        return sorted(seen)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._cam_ids:
            return

        legend_ids = self._object_ids_present()
        legend_top = self.height() - _LEGEND_HEIGHT if legend_ids else self.height()

        n = len(self._cam_ids)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        grid_w = self.width()
        grid_h = legend_top
        cell_w = grid_w / cols
        cell_h = grid_h / rows

        for i, cam_id in enumerate(self._cam_ids):
            row = i // cols
            col = i % cols
            x = col * cell_w
            y = row * cell_h
            self._draw_panel(painter, cam_id, x, y, cell_w, cell_h)

        if legend_ids:
            self._draw_legend(painter, legend_ids, legend_top)

    def _draw_panel(
        self,
        painter: QPainter,
        cam_id: int,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        inner_x = x + _PANEL_PADDING
        inner_y = y + _PANEL_PADDING
        inner_w = w - 2 * _PANEL_PADDING
        inner_h = h - 2 * _PANEL_PADDING
        if inner_w <= 0 or inner_h <= 0:
            return

        painter.setPen(QPen(QColor(90, 90, 90)))
        painter.setBrush(QColor(28, 28, 28))
        painter.drawRect(int(inner_x), int(inner_y), int(inner_w), int(inner_h))

        painter.setPen(QColor(210, 210, 210))
        painter.setFont(QFont("Monospace", 8))
        painter.drawText(
            int(inner_x),
            int(inner_y),
            int(inner_w),
            _TITLE_HEIGHT,
            Qt.AlignmentFlag.AlignCenter,
            f"Cam {cam_id}",
        )

        image_top = inner_y + _TITLE_HEIGHT
        image_h = inner_h - _TITLE_HEIGHT
        if image_h <= 0:
            return

        cam_w, cam_h = self._cam_sizes.get(cam_id, (0, 0))
        if cam_w <= 0 or cam_h <= 0:
            return

        # Fit the image rectangle into the panel preserving aspect ratio.
        scale = min(inner_w / cam_w, image_h / cam_h)
        draw_w = cam_w * scale
        draw_h = cam_h * scale
        origin_x = inner_x + (inner_w - draw_w) / 2
        origin_y = image_top + (image_h - draw_h) / 2

        points = self._points_by_cam.get(cam_id, [])
        if not points:
            painter.setPen(QColor(110, 110, 110))
            painter.drawText(
                int(inner_x),
                int(image_top),
                int(inner_w),
                int(image_h),
                Qt.AlignmentFlag.AlignCenter,
                "no view",
            )
            return

        painter.setPen(Qt.PenStyle.NoPen)
        for img_x, img_y, obj_id in points:
            color = _FACE_COLORS.get(obj_id, _FALLBACK_COLOR)
            painter.setBrush(color)
            center = QPointF(origin_x + img_x * scale, origin_y + img_y * scale)
            painter.drawEllipse(center, _POINT_RADIUS, _POINT_RADIUS)

    def _draw_legend(self, painter: QPainter, object_ids: list[int], top: float) -> None:
        painter.setFont(QFont("Monospace", 8))
        swatch = 10
        gap = 6
        x = _PANEL_PADDING
        y = top + (_LEGEND_HEIGHT - swatch) / 2

        for obj_id in object_ids:
            color = _FACE_COLORS.get(obj_id, _FALLBACK_COLOR)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(int(x), int(y), swatch, swatch)
            x += swatch + gap

            label = _FACE_LABELS.get(obj_id, f"obj {obj_id}")
            painter.setPen(QColor(210, 210, 210))
            text_w = painter.fontMetrics().horizontalAdvance(label)
            painter.drawText(int(x), int(top), text_w + gap, _LEGEND_HEIGHT, Qt.AlignmentFlag.AlignVCenter, label)
            x += text_w + 3 * gap
