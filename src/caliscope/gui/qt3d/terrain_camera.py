"""Terrain-style camera controller for Qt3D scenes.

Provides map-like orbit/pan/zoom interaction around a fixed focus point
with Z always up — suitable for inspecting a mocap capture volume.
"""

import math

from PySide6.Qt3DRender import Qt3DRender
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QVector3D
from PySide6.QtGui import QMouseEvent, QWheelEvent


class TerrainCameraController:
    """Map-like camera that orbits around a focus point with Z always up.

    Controls:
    - Left drag: orbit (azimuth + elevation, Z-up locked)
    - Right drag / middle drag: pan (shift focus point in screen plane)
    - Scroll: zoom (move closer/further from focus)

    Elevation is clamped to [5, 85] degrees to prevent gimbal lock
    and to keep the view sensible for a mocap capture volume.
    """

    def __init__(self, camera: Qt3DRender.QCamera):
        self._camera = camera

        # Spherical coordinates relative to focus point
        self._azimuth = math.radians(45)  # horizontal angle
        self._elevation = math.radians(35)  # angle above horizon
        self._distance = 6.0  # distance from focus
        self._focus = QVector3D(0, 0, 0.5)  # look-at point

        # Sensitivity
        self._orbit_speed = 0.3  # degrees per pixel
        self._pan_speed = 0.005  # world units per pixel (scaled by distance)
        self._zoom_speed = 0.001  # fraction per scroll degree

        # Mouse state
        self._last_pos = QPoint()
        self._dragging_orbit = False
        self._dragging_pan = False

        self._apply()

    def _apply(self) -> None:
        """Recompute camera position from spherical coords and apply."""
        self._elevation = max(math.radians(5), min(math.radians(85), self._elevation))
        self._distance = max(0.1, self._distance)

        cos_el = math.cos(self._elevation)
        eye = QVector3D(
            self._distance * cos_el * math.cos(self._azimuth),
            self._distance * cos_el * math.sin(self._azimuth),
            self._distance * math.sin(self._elevation),
        )

        self._camera.setPosition(self._focus + eye)
        self._camera.setViewCenter(self._focus)
        self._camera.setUpVector(QVector3D(0, 0, 1))

    def mouse_press(self, event: QMouseEvent) -> None:
        self._last_pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_orbit = True
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._dragging_pan = True

    def mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_orbit = False
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._dragging_pan = False

    def mouse_move(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos

        if self._dragging_orbit:
            self._azimuth -= math.radians(dx * self._orbit_speed)
            self._elevation += math.radians(dy * self._orbit_speed)
            self._apply()

        elif self._dragging_pan:
            right = QVector3D(
                -math.sin(self._azimuth),
                math.cos(self._azimuth),
                0,
            )
            forward = QVector3D(
                math.cos(self._elevation) * math.cos(self._azimuth),
                math.cos(self._elevation) * math.sin(self._azimuth),
                math.sin(self._elevation),
            )
            up = QVector3D.crossProduct(right, forward).normalized()

            pan_scale = self._pan_speed * self._distance
            self._focus -= right * (dx * pan_scale)
            self._focus -= up * (dy * pan_scale)
            self._apply()

    def wheel(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.0 - delta * self._zoom_speed
        self._distance *= factor
        self._apply()

    def set_focus(self, focus: QVector3D) -> None:
        self._focus = focus
        self._apply()

    def set_distance(self, distance: float) -> None:
        self._distance = distance
        self._apply()

    @property
    def azimuth(self) -> float:
        return self._azimuth

    @property
    def elevation(self) -> float:
        return self._elevation

    @property
    def distance(self) -> float:
        return self._distance

    @property
    def focus(self) -> QVector3D:
        return QVector3D(self._focus)  # return a copy
