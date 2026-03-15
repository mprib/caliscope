"""
Qt3D-based 3D visualization widget for triangulated motion capture playback.

Replaces PlaybackVizWidget (PyVista/VTK) with a pure Qt3D render pipeline.
Qt3D's render loop is event-driven, eliminating the idle CPU overhead of
VTK's polling interactor.

Displays camera frustums and animated 3D point positions from WorldPoints data,
with playback controls for play/pause, looping, speed adjustment, and frame scrubbing.
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QImage, QVector3D
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.Qt3DRender import Qt3DRender

from caliscope import ICONS_DIR
from caliscope.gui.qt3d.primitives import (
    SphereCloud,
    build_floor_grid,
    build_origin_axes,
    create_double_sided_mesh,
    create_line_entity,
    numpy_to_qbytearray,
)
from caliscope.gui.qt3d.terrain_camera import TerrainCameraController
from caliscope.gui.view_models.playback_view_model import PlaybackViewModel

logger = logging.getLogger(__name__)


def _icon(name: str) -> QIcon:
    """Load SVG icon from gui/icons directory."""
    return QIcon(str(ICONS_DIR / f"{name}.svg"))


class Qt3DPlaybackWidget(QWidget):
    """
    Qt3D-based widget for animated playback of triangulated 3D points.

    Replaces PlaybackVizWidget (PyVista) with an event-driven Qt3D render
    pipeline. No idle CPU cost — Qt3D only renders when the scene changes.

    Features:
    - Play/pause animation with configurable speed (0.1x to 3.0x)
    - Loop toggle for continuous playback
    - Frame slider for manual scrubbing
    - Camera label visibility toggle (overlay stub, deferred)
    - QRenderCapture-based screenshot support

    Note: Inherits from QWidget (not QMainWindow) so it can be embedded in layouts.
    """

    # Default camera frustum scale for production scenes (meters, small scale)
    DEFAULT_CAMERA_SCALE = 0.0002

    def __init__(
        self,
        view_model: PlaybackViewModel,
        parent: QWidget | None = None,
        camera_scale: float | None = None,
    ):
        super().__init__(parent)

        self.view_model = view_model
        self._camera_scale = camera_scale if camera_scale is not None else self.DEFAULT_CAMERA_SCALE
        self.sync_index: int = self.view_model.min_index

        # UI state
        self.show_camera_labels = True
        self.is_playing = False
        self.loop_enabled = True
        self.speed_multiplier = 1.0

        # Dynamic scene object references (None until _create_dynamic_geometry runs)
        self._sphere_cloud: SphereCloud | None = None
        self._wire_entity: Qt3DCore.QEntity | None = None
        self._wire_buffer: Qt3DCore.QBuffer | None = None
        self._wire_indices: np.ndarray | None = None

        # Label anchor positions for future 2D overlay
        self._label_anchors: list | None = None

        # Cache icons for play/pause toggle
        self._play_icon = _icon("play")
        self._pause_icon = _icon("pause")

        # Timer for animation
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame)

        # --- Main layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Qt3DWindow ---
        self._view = Qt3DExtras.Qt3DWindow()
        self._view.defaultFrameGraph().setClearColor(QColor(25, 25, 25))

        # QRenderCapture must be a child of the active frame graph node so it
        # intercepts the render pipeline. widget.grab() cannot capture Qt3D
        # content because Qt3DWindow renders to its own GPU surface via
        # createWindowContainer.
        self._render_capture = Qt3DRender.QRenderCapture(self._view.activeFrameGraph())

        # Forward mouse/wheel events from the Qt3DWindow to our terrain controller.
        # Qt3DWindow is a QWindow (not a QWidget), so we install a filter on it.
        self._view.installEventFilter(self)

        container = QWidget.createWindowContainer(self._view, self)
        main_layout.addWidget(container, stretch=1)

        # --- Controls bar ---
        self._control_bar = self._create_controls()
        main_layout.addWidget(self._control_bar)

        # --- Scene setup ---
        self._root = Qt3DCore.QEntity()
        self._setup_camera()
        self._create_static_geometry()
        self._create_dynamic_geometry()
        self._on_sync_index_changed(self.sync_index)
        self._view.setRootEntity(self._root)

    # -------------------------------------------------------------------------
    # Camera setup
    # -------------------------------------------------------------------------

    def _setup_camera(self) -> None:
        """Configure perspective camera and terrain controller."""
        camera = self._view.camera()
        camera.lens().setPerspectiveProjection(45.0, 16.0 / 9.0, 0.01, 100.0)
        self._cam_controller = TerrainCameraController(camera)
        self._set_adaptive_camera()

    def _set_adaptive_camera(self) -> None:
        """Position camera based on scene extent derived from camera positions."""
        positions = self.view_model.get_camera_positions()
        if positions is None or len(positions) == 0:
            self._cam_controller.set_focus(QVector3D(0, 0, 0))
            self._cam_controller.set_distance(6.0)
            return

        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)
        center = (min_coords + max_coords) / 2
        extent = max_coords - min_coords
        max_extent = float(max(extent))

        self._cam_controller.set_focus(QVector3D(float(center[0]), float(center[1]), float(center[2])))
        self._cam_controller.set_distance(max(max_extent * 2.0, 0.5))

    # -------------------------------------------------------------------------
    # Event filter — forward mouse/wheel events to terrain controller
    # -------------------------------------------------------------------------

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if obj is self._view:
            event_type = event.type()
            if event_type == QEvent.Type.MouseButtonPress:
                self._cam_controller.mouse_press(event)
                return True
            elif event_type == QEvent.Type.MouseButtonRelease:
                self._cam_controller.mouse_release(event)
                return True
            elif event_type == QEvent.Type.MouseMove:
                self._cam_controller.mouse_move(event)
                return True
            elif event_type == QEvent.Type.Wheel:
                self._cam_controller.wheel(event)
                return True
        return super().eventFilter(obj, event)

    # -------------------------------------------------------------------------
    # Scene construction
    # -------------------------------------------------------------------------

    def _create_static_geometry(self) -> None:
        """Build camera frustums, floor grid, and origin axes."""
        camera_geom = self.view_model.get_camera_geometry(scale=self._camera_scale)
        if camera_geom is not None:
            vertices = camera_geom["vertices"].astype(np.float32)

            # Solid faces (dark green, double-sided)
            create_double_sided_mesh(
                vertices,
                camera_geom["triangles"].flatten().astype(np.uint32),
                QColor(30, 120, 30),
                self._root,
            )

            # Bright edge wireframe
            create_line_entity(
                vertices,
                camera_geom["edges"].flatten().astype(np.uint32),
                QColor(80, 255, 80),
                self._root,
            )

            # Store label anchors for future 2D overlay
            self._label_anchors = camera_geom["labels"]

        # Floor grid and axes — size based on camera spread
        positions = self.view_model.get_camera_positions()
        if positions is not None and len(positions) > 0:
            distances = np.sqrt((positions[:, :2] ** 2).sum(axis=1))
            grid_size = float(max(distances.max() * 2.5, 0.5))
        else:
            grid_size = 5.0

        build_floor_grid(self._root, size=grid_size)
        build_origin_axes(self._root)

    def _create_dynamic_geometry(self) -> None:
        """Create sphere cloud and wireframe line entity for per-frame updates."""
        if self.view_model.n_points == 0:
            logger.info("Camera-only mode: skipping dynamic geometry")
            return

        frame_geom = self.view_model.get_frame_geometry(self.sync_index)
        lines, line_colors = self.view_model.get_static_wireframe_data()

        # Sphere cloud — one sphere per tracked point
        self._sphere_cloud = SphereCloud(
            n_points=self.view_model.n_points,
            color=QColor(200, 200, 200),
            parent=self._root,
        )
        self._sphere_cloud.update_positions(frame_geom.points)

        # Wireframe connecting the spheres
        if len(lines) > 0:
            wire_indices = lines.flatten().astype(np.uint32)
            self._wire_entity, self._wire_buffer = create_line_entity(
                frame_geom.points,
                wire_indices,
                QColor(100, 180, 255),
                self._root,
            )
            self._wire_indices = wire_indices
        else:
            self._wire_entity = None
            self._wire_buffer = None

    # -------------------------------------------------------------------------
    # Per-frame update
    # -------------------------------------------------------------------------

    def _on_sync_index_changed(self, sync_index: int) -> None:
        """Update dynamic geometry positions for the given sync index."""
        self.sync_index = sync_index

        if self._sphere_cloud is None:
            return

        frame_geom = self.view_model.get_frame_geometry(sync_index)
        self._sphere_cloud.update_positions(frame_geom.points)

        if self._wire_buffer is not None:
            self._wire_buffer.setData(numpy_to_qbytearray(frame_geom.points))

    # -------------------------------------------------------------------------
    # Controls bar
    # -------------------------------------------------------------------------

    def _create_controls(self) -> QWidget:
        """Create playback control bar with play/loop/speed controls and slider."""
        controls = QWidget(self)
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(5, 5, 5, 5)

        # Play/Pause button
        self.play_button = QPushButton(self)
        self.play_button.setIcon(self._play_icon)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setFixedSize(30, 30)
        self.play_button.setToolTip("Play/Pause (toggle)")
        layout.addWidget(self.play_button)

        # Loop toggle button
        self.loop_button = QPushButton(self)
        self.loop_button.setIcon(_icon("repeat"))
        self.loop_button.setCheckable(True)
        self.loop_button.setChecked(True)
        self.loop_button.clicked.connect(self._toggle_loop)
        self.loop_button.setFixedSize(30, 30)
        self.loop_button.setToolTip("Loop playback (toggle)")
        layout.addWidget(self.loop_button)

        # Speed slider
        self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(30)
        self.speed_slider.setValue(10)
        self.speed_slider.setFixedWidth(80)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self.speed_slider.setToolTip("Playback speed")
        layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("1.0x", self)
        self.speed_label.setFixedWidth(35)
        layout.addWidget(self.speed_label)

        layout.addStretch()

        # Camera labels toggle (wired to stub — overlay deferred)
        self.labels_checkbox = QCheckBox("Camera Labels", self)
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self._on_labels_toggled)
        layout.addWidget(self.labels_checkbox)

        # Frame slider
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(self.view_model.min_index)
        self.slider.setMaximum(self.view_model.max_index)
        self.slider.setValue(self.sync_index)
        self.slider.valueChanged.connect(self._on_sync_index_changed)
        layout.addWidget(self.slider, stretch=1)

        return controls

    # -------------------------------------------------------------------------
    # Playback control callbacks
    # -------------------------------------------------------------------------

    def _toggle_playback(self, checked: bool) -> None:
        self.is_playing = checked
        if checked:
            self.play_button.setIcon(self._pause_icon)
            self._start_playback()
        else:
            self.play_button.setIcon(self._play_icon)
            self.playback_timer.stop()

    def _start_playback(self) -> None:
        if self.view_model.frame_rate <= 0:
            return
        interval_ms = int(1000 / (self.view_model.frame_rate * self.speed_multiplier))
        self.playback_timer.start(interval_ms)

    def _advance_frame(self) -> None:
        next_index = self.sync_index + 1
        if next_index > self.view_model.max_index:
            if self.loop_enabled:
                next_index = self.view_model.min_index
            else:
                self.play_button.setChecked(False)
                self.playback_timer.stop()
                return
        self.slider.setValue(next_index)

    def _toggle_loop(self, checked: bool) -> None:
        self.loop_enabled = checked
        self.loop_button.setChecked(checked)

    def _on_speed_changed(self, value: int) -> None:
        self.speed_multiplier = value / 10.0
        self.speed_label.setText(f"{self.speed_multiplier:.1f}x")
        if self.is_playing:
            self._start_playback()

    def _on_labels_toggled(self, state: int) -> None:
        # Stub — labels not yet implemented as 2D screen-space overlays
        self.show_camera_labels = state == Qt.CheckState.Checked.value

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_view_model(self, view_model: PlaybackViewModel, preserve_camera: bool = False) -> None:
        """Replace the ViewModel and rebuild the entire scene.

        Used when switching recordings or trackers in post-processing.
        Rebuilds the full scene (not just dynamic geometry) because cameras
        may differ between recordings.

        The QRenderCapture lives on the frame graph (not the entity graph),
        so it survives root entity replacement without any additional action.

        Args:
            view_model: New view model to display.
            preserve_camera: If True, restore camera position and sync_index
                after rebuild (useful for coordinate transforms).
        """
        logger.info(f"Switching view model: {self.view_model.n_points} -> {view_model.n_points} points")

        # Save camera state before clearing
        saved_cam: dict | None = None
        saved_sync: int | None = None
        if preserve_camera:
            saved_cam = {
                "azimuth": self._cam_controller.azimuth,
                "elevation": self._cam_controller.elevation,
                "distance": self._cam_controller.distance,
                "focus": self._cam_controller.focus,
            }
            saved_sync = self.sync_index

        # Stop playback
        self.playback_timer.stop()
        self.is_playing = False
        self.play_button.setChecked(False)
        self.play_button.setIcon(self._play_icon)

        # Clear dynamic state references before slider updates to prevent
        # _on_sync_index_changed from operating on stale/mismatched geometry
        self._sphere_cloud = None
        self._wire_entity = None
        self._wire_buffer = None

        self.view_model = view_model

        # Determine sync index for the new view model
        if saved_sync is not None:
            self.sync_index = max(view_model.min_index, min(saved_sync, view_model.max_index))
        else:
            self.sync_index = view_model.min_index

        # Update slider range (setValue may trigger _on_sync_index_changed,
        # but dynamic refs are already None so it returns early safely)
        self.slider.setMinimum(view_model.min_index)
        self.slider.setMaximum(view_model.max_index)
        self.slider.setValue(self.sync_index)

        # Rebuild: create a new root entity, attach everything to it, then
        # swap it in. The old root's Qt3D subtree is cleaned up by Qt.
        self._root = Qt3DCore.QEntity()
        self._setup_camera()
        self._create_static_geometry()
        self._create_dynamic_geometry()
        self._on_sync_index_changed(self.sync_index)
        self._view.setRootEntity(self._root)

        # Restore camera orientation if requested
        if saved_cam is not None:
            self._cam_controller._azimuth = saved_cam["azimuth"]
            self._cam_controller._elevation = saved_cam["elevation"]
            self._cam_controller._distance = saved_cam["distance"]
            self._cam_controller._focus = saved_cam["focus"]
            self._cam_controller._apply()

    def set_sync_index(self, sync_index: int) -> None:
        """Set frame index programmatically (for external slider control).

        Use this when embedding the widget in a container with a shared slider.
        """
        self.sync_index = sync_index
        self._on_sync_index_changed(sync_index)

        # Update internal slider without re-triggering _on_sync_index_changed
        self.slider.blockSignals(True)
        self.slider.setValue(sync_index)
        self.slider.blockSignals(False)

    def show_playback_controls(self, visible: bool) -> None:
        """Show or hide the playback control bar.

        Use this when embedding the widget in a container with shared controls.

        Args:
            visible: If False, hides the slider, play/pause button, speed control, etc.
        """
        self._control_bar.setVisible(visible)

    def suspend_vtk(self) -> None:
        """No-op. Qt3D's render loop is event-driven.

        Unlike VTK's polling interactor (which runs a ~10ms timer continuously),
        Qt3D only renders when the scene is marked dirty. There is no idle timer
        to suspend.
        """
        pass

    def resume_vtk(self) -> None:
        """No-op. Qt3D's render loop is event-driven.

        Unlike VTK's polling interactor, Qt3D requires no explicit resume —
        the next scene change will trigger a render automatically.
        """
        pass

    def capture_screenshot(self) -> QImage | None:
        """Capture the Qt3D scene via QRenderCapture.

        widget.grab() cannot capture Qt3D content because Qt3DWindow renders
        to its own GPU surface via createWindowContainer. QRenderCapture
        intercepts the render pipeline directly and returns the framebuffer
        contents as a QImage.

        Returns:
            QImage on success, None if the capture does not complete within
            500ms (e.g., render pipeline not yet initialised).
        """
        from PySide6.QtCore import QEventLoop

        reply = self._render_capture.requestCapture()

        loop = QEventLoop()
        reply.completed.connect(loop.quit)
        QTimer.singleShot(500, loop.quit)
        loop.exec()

        if reply.isComplete():
            return reply.image()
        return None

    # -------------------------------------------------------------------------
    # Camera label overlay stub
    # -------------------------------------------------------------------------

    def _update_label_overlays(self) -> None:
        """Update 2D screen-space camera labels.

        TODO: Project 3D label anchor positions to 2D screen coordinates
        and position QLabel widgets accordingly. Deferred — need to solve
        world-to-screen projection via camera's viewProjectionMatrix.
        """
        pass
