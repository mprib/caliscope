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
import math
from typing import cast

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QMouseEvent, QVector3D, QWheelEvent
from PySide6.QtWidgets import (
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

from caliscope.gui import ICONS_DIR
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

# Default sphere radius in meters (12mm). Used as the 1.0x baseline
# for the sphere size slider.
_DEFAULT_SPHERE_RADIUS = 0.012


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
    - Adjustable camera frustum size, floor grid size, and sphere size via sliders
    - QRenderCapture-based screenshot support

    Note: Inherits from QWidget (not QMainWindow) so it can be embedded in layouts.
    """

    camera_size_multiplier_changed = Signal(float)
    grid_size_multiplier_changed = Signal(float)
    sphere_size_multiplier_changed = Signal(float)

    def __init__(
        self,
        view_model: PlaybackViewModel,
        parent: QWidget | None = None,
        camera_scale: float | None = None,
        camera_size_multiplier: float = 1.0,
        grid_size_multiplier: float = 1.0,
        sphere_size_multiplier: float = 1.0,
    ):
        super().__init__(parent)

        self.view_model = view_model
        # If caller provides an explicit scale, use it (e.g., synthetic tests).
        # Otherwise, computed adaptively from scene extent in _set_adaptive_camera().
        self._camera_scale_override = camera_scale
        self._camera_scale: float = 0.0  # Set by _set_adaptive_camera or override
        self._default_camera_scale: float = 0.0  # Adaptive baseline (slider 1.0x)
        self._default_grid_size: float = 5.0  # Adaptive baseline (slider 1.0x)
        self._camera_scale_multiplier: float = camera_size_multiplier  # Current slider multiplier
        self._grid_size_multiplier: float = grid_size_multiplier  # Current slider multiplier
        self._sphere_size_multiplier: float = sphere_size_multiplier  # Current slider multiplier
        self.sync_index: int = self.view_model.min_index

        # UI state
        self.is_playing = False
        self.loop_enabled = True
        self.speed_multiplier = 1.0

        # Dynamic scene object references (None until _create_dynamic_geometry runs)
        self._sphere_cloud: SphereCloud | None = None
        self._wire_entity: Qt3DCore.QEntity | None = None
        self._wire_buffer: Qt3DCore.QBuffer | None = None
        self._wire_indices: np.ndarray | None = None

        # Container entities for selective rebuilding — cameras and grid can be
        # rebuilt independently when their slider changes, without touching the
        # dynamic point cloud or each other.
        self._camera_container: Qt3DCore.QEntity | None = None
        self._grid_container: Qt3DCore.QEntity | None = None

        # Prevent shiboken6 GC of Qt3D entities still referenced by the C++ scene graph.
        # Shiboken destroys C++ objects when Python wrappers lose all references, even
        # when the entity has a Qt parent. Qt3D's render thread holds internal refs to
        # scene entities asynchronously — if shiboken destroys one mid-traversal, the
        # result is use-after-free. Append every QEntity here at creation time. Never
        # remove entries — the memory cost is negligible (kilobytes per scene swap).
        self._retained_entities: list[Qt3DCore.QEntity] = []

        # Debounce timers for slider-driven geometry rebuilds.
        # Without debouncing, dragging a slider fires valueChanged on every
        # pixel of movement, triggering ~10-20 full geometry rebuilds per drag.
        self._cam_rebuild_timer = QTimer(self)
        self._cam_rebuild_timer.setSingleShot(True)
        self._cam_rebuild_timer.setInterval(50)
        self._cam_rebuild_timer.timeout.connect(self._create_camera_geometry)
        self._grid_rebuild_timer = QTimer(self)
        self._grid_rebuild_timer.setSingleShot(True)
        self._grid_rebuild_timer.setInterval(50)
        self._grid_rebuild_timer.timeout.connect(self._create_grid_geometry)

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

        # QRenderCapture is created lazily in capture_screenshot() because
        # merely attaching it to the frame graph crashes under xvfb/software
        # rendering (Mesa llvmpipe). The RHI OpenGL backend segfaults in the
        # render thread when QRenderCapture tries to read back framebuffer
        # pixels from a virtual framebuffer. Deferring creation means headless
        # tests never trigger the crash.
        self._render_capture: Qt3DRender.QRenderCapture | None = None

        # Forward mouse/wheel events from the Qt3DWindow to our terrain controller.
        # Qt3DWindow is a QWindow (not a QWidget), so we install a filter on it.
        self._view.installEventFilter(self)

        container = QWidget.createWindowContainer(self._view, self)
        main_layout.addWidget(container, stretch=1)

        # --- Controls bars ---
        # Playback and appearance are separate widgets so that
        # show_playback_controls(False) can hide play/pause/slider
        # without hiding the camera-size / grid-size sliders.
        self._playback_bar = self._create_playback_controls()
        self._appearance_bar = self._create_appearance_controls()
        main_layout.addWidget(self._playback_bar)
        main_layout.addWidget(self._appearance_bar)

        # --- Scene setup ---
        # Root entity is created once and set once. setRootEntity() is a one-shot
        # initialization — calling it again while the render thread is mid-traversal
        # causes use-after-free segfaults. All scene content lives in a swappable
        # _scene container child of root.
        self._root = Qt3DCore.QEntity()
        self._scene: Qt3DCore.QEntity | None = None
        self._build_scene()
        self._view.setRootEntity(self._root)

    # -------------------------------------------------------------------------
    # Scene lifecycle
    # -------------------------------------------------------------------------

    def _build_scene(self) -> None:
        """Build a fresh scene container under the permanent root entity.

        If an old scene container exists, it is disabled immediately (safe for
        the render thread — setEnabled is a batched property change). The old
        scene is appended to _retained_entities so shiboken6 never destroys it.
        """
        if self._scene is not None:
            self._scene.setEnabled(False)
            self._retained_entities.append(self._scene)

        # Build new scene as child of root — Qt3D adds it to the render tree
        # automatically when we set its parent
        self._scene = Qt3DCore.QEntity(self._root)

        self._setup_camera()
        self._retained_entities.extend(build_origin_axes(self._scene))
        self._create_camera_geometry()
        self._create_grid_geometry()
        self._create_dynamic_geometry()
        self._on_sync_index_changed(self.sync_index)

        logger.info(
            "Scene built: camera_container=%s, grid_container=%s, sphere_cloud=%s, wire_entity=%s, scene_enabled=%s",
            self._camera_container is not None,
            self._grid_container is not None,
            self._sphere_cloud is not None,
            self._wire_entity is not None,
            self._scene.isEnabled() if self._scene else "N/A",
        )

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
        """Position camera and compute adaptive defaults from scene extent.

        Derives sensible defaults for camera frustum scale and grid size
        from the spatial distribution of cameras. These defaults serve as
        the 1.0x baseline for the user-adjustable sliders.
        """
        positions = self.view_model.get_camera_positions()
        if positions is None or len(positions) == 0:
            self._default_camera_scale = 0.0005
            self._default_grid_size = 5.0
            self._camera_scale = self._camera_scale_override or self._default_camera_scale
            self._cam_controller.set_focus(QVector3D(0, 0, 0))
            self._cam_controller.set_distance(6.0)
            return

        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)
        center = (min_coords + max_coords) / 2
        extent = max_coords - min_coords
        max_extent = float(max(extent))
        scene_extent = max(max_extent, 0.5)

        # Position the viewing camera
        self._cam_controller.set_focus(QVector3D(float(center[0]), float(center[1]), float(center[2])))
        view_distance = max(scene_extent * 2.0, 0.5)
        self._cam_controller.set_distance(view_distance)

        # Adapt near/far clipping planes to scene scale.
        # The far plane must contain the entire scene from the viewing distance.
        # Near plane uses a 1:10000 ratio to avoid z-fighting while keeping
        # close geometry visible.
        far = max(view_distance * 5.0, 100.0)
        near = far * 0.0001
        self._view.camera().lens().setPerspectiveProjection(45.0, 16.0 / 9.0, near, far)
        logger.info(f"Clipping planes: near={near:.6f}, far={far:.1f}")

        # Adaptive camera frustum scale: make frustum depth ~5% of scene extent.
        # build_camera_geometry computes depth as focal_length * scale, and typical
        # focal lengths are ~1000px, so scale ~ scene_extent * 5e-5.
        self._default_camera_scale = scene_extent * 5e-5
        logger.info(
            f"Scene extent={scene_extent:.2f}m, "
            f"default camera scale={self._default_camera_scale:.6f}, "
            f"view distance={view_distance:.2f}m, far={far:.1f}"
        )

        # Override takes precedence (e.g., synthetic storyboard with known scale)
        if self._camera_scale_override is not None:
            self._camera_scale = self._camera_scale_override
        else:
            self._camera_scale = self._default_camera_scale * self._camera_scale_multiplier

        # Adaptive grid size: cover the camera spread with margin
        distances = np.sqrt((positions[:, :2] ** 2).sum(axis=1))
        self._default_grid_size = float(max(distances.max() * 2.5, 0.5))

    # -------------------------------------------------------------------------
    # Event filter — forward mouse/wheel events to terrain controller
    # -------------------------------------------------------------------------

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._view:
            # Guard: controller may not exist yet during __init__ (event filter
            # is installed before _build_scene creates the controller), or
            # during set_view_model when the scene is being rebuilt.
            if not hasattr(self, "_cam_controller"):
                return True

            event_type = event.type()
            if event_type == QEvent.Type.MouseButtonPress:
                self._cam_controller.mouse_press(cast(QMouseEvent, event))
                return True
            elif event_type == QEvent.Type.MouseButtonRelease:
                self._cam_controller.mouse_release(cast(QMouseEvent, event))
                return True
            elif event_type == QEvent.Type.MouseMove:
                self._cam_controller.mouse_move(cast(QMouseEvent, event))
                return True
            elif event_type == QEvent.Type.Wheel:
                self._cam_controller.wheel(cast(QWheelEvent, event))
                return True
            elif event_type == QEvent.Type.MouseButtonDblClick:
                # Consume double-clicks to prevent Qt3D from attempting
                # object picking, which can crash without a pick handler.
                return True
        return super().eventFilter(obj, event)

    # -------------------------------------------------------------------------
    # Scene construction — static geometry (cameras, grid, axes)
    # -------------------------------------------------------------------------

    def _create_camera_geometry(self) -> None:
        """Build camera frustum entities under a fresh container.

        A new container is created each rebuild. The old container is disabled
        and retained for shiboken GC safety.
        """
        if self._camera_container is not None:
            self._camera_container.setEnabled(False)

        assert self._scene is not None
        self._camera_container = Qt3DCore.QEntity(self._scene)
        self._retained_entities.append(self._camera_container)

        camera_geom = self.view_model.get_camera_geometry(scale=self._camera_scale)
        logger.info(
            "Camera geometry: scale=%.6f, geom=%s",
            self._camera_scale,
            f"{len(camera_geom['vertices'])} verts" if camera_geom else "None",
        )
        if camera_geom is not None:
            vertices = camera_geom["vertices"].astype(np.float32)

            # Solid faces (dark green, double-sided)
            face_entity = create_double_sided_mesh(
                vertices,
                camera_geom["triangles"].flatten().astype(np.uint32),
                QColor(60, 180, 60),
                self._camera_container,
            )
            self._retained_entities.append(face_entity)

            # Bright edge wireframe
            edge_entity, _ = create_line_entity(
                vertices,
                camera_geom["edges"].flatten().astype(np.uint32),
                QColor(120, 255, 120),
                self._camera_container,
            )
            self._retained_entities.append(edge_entity)

    def _create_grid_geometry(self) -> None:
        """Build floor grid entities under a fresh container.

        A new container is created each rebuild. The old container is disabled
        and retained for shiboken GC safety.
        """
        if self._grid_container is not None:
            self._grid_container.setEnabled(False)

        assert self._scene is not None
        self._grid_container = Qt3DCore.QEntity(self._scene)
        self._retained_entities.append(self._grid_container)

        grid_size = self._default_grid_size * self._grid_size_multiplier
        grid_entities = build_floor_grid(self._grid_container, size=grid_size)
        self._retained_entities.extend(grid_entities)

    def _create_dynamic_geometry(self) -> None:
        """Create sphere cloud and wireframe line entity for per-frame updates."""
        if self.view_model.n_points == 0:
            logger.info("Camera-only mode: skipping dynamic geometry")
            return

        frame_geom = self.view_model.get_frame_geometry(self.sync_index)
        lines, line_colors = self.view_model.get_static_wireframe_data()

        # Sphere cloud — one sphere per tracked point
        assert self._scene is not None
        self._sphere_cloud = SphereCloud(
            n_points=self.view_model.n_points,
            color=QColor(240, 240, 240),
            parent=self._scene,
            sphere_radius=_DEFAULT_SPHERE_RADIUS * self._sphere_size_multiplier,
        )
        self._retained_entities.append(self._sphere_cloud._parent_entity)  # retain for shiboken GC safety
        self._sphere_cloud.update_positions(frame_geom.points)

        # Wireframe connecting the spheres
        if len(lines) > 0:
            wire_indices = lines.flatten().astype(np.uint32)
            self._wire_entity, self._wire_buffer = create_line_entity(
                frame_geom.points,
                wire_indices,
                QColor(140, 210, 255),
                self._scene,
            )
            self._retained_entities.append(self._wire_entity)  # retain for shiboken GC safety
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
            # Sanitize NaN positions before pushing to the GPU. Missing points
            # are NaN in the frame geometry. Feeding NaN floats into a vertex
            # buffer causes undefined GPU rasterizer behavior — lines connecting
            # NaN vertices can corrupt the depth buffer, making other geometry
            # (cameras, grid) fail the depth test and vanish. Moving missing
            # vertices far off-screen produces zero-length degenerate lines
            # that the GPU safely discards.
            sanitized = frame_geom.points.copy()
            nan_mask = np.isnan(sanitized[:, 0])
            sanitized[nan_mask] = 99999.0
            self._wire_buffer.setData(numpy_to_qbytearray(sanitized))

    # -------------------------------------------------------------------------
    # Controls bar
    # -------------------------------------------------------------------------

    def _create_playback_controls(self) -> QWidget:
        """Create playback control bar with play/loop/speed controls and slider."""
        bar = QWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(5, 2, 5, 2)

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

        # Frame slider
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(self.view_model.min_index)
        self.slider.setMaximum(self.view_model.max_index)
        self.slider.setValue(self.sync_index)
        self.slider.valueChanged.connect(self._on_sync_index_changed)
        layout.addWidget(self.slider, stretch=1)

        return bar

    def _create_appearance_controls(self) -> QWidget:
        """Create scene appearance bar with camera-size, grid-size, and sphere-size sliders.

        This is a separate widget from the playback bar so that
        show_playback_controls(False) can hide play/pause without
        also hiding these appearance sliders.
        """
        bar = QWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(5, 2, 5, 2)

        # Camera size slider — logarithmic mapping for intuitive control.
        # Slider position 50 = 1.0x (default), 0 = 0.1x, 100 = 10x.
        layout.addWidget(QLabel("Cam:", self))
        self._cam_size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._cam_size_slider.setMinimum(0)
        self._cam_size_slider.setMaximum(100)
        self._cam_size_slider.setValue(self._multiplier_to_slider(self._camera_scale_multiplier))
        self._cam_size_slider.setFixedWidth(100)
        self._cam_size_slider.setToolTip("Camera frustum size (1.0x = adaptive default)")
        self._cam_size_slider.valueChanged.connect(self._on_cam_size_changed)
        self._cam_size_slider.sliderReleased.connect(self._on_cam_size_released)
        layout.addWidget(self._cam_size_slider)
        self._cam_size_label = QLabel(f"{self._camera_scale_multiplier:.1f}x", self)
        self._cam_size_label.setFixedWidth(35)
        layout.addWidget(self._cam_size_label)

        layout.addSpacing(20)

        # Grid size slider — same logarithmic mapping
        layout.addWidget(QLabel("Grid:", self))
        self._grid_size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._grid_size_slider.setMinimum(0)
        self._grid_size_slider.setMaximum(100)
        self._grid_size_slider.setValue(self._multiplier_to_slider(self._grid_size_multiplier))
        self._grid_size_slider.setFixedWidth(100)
        self._grid_size_slider.setToolTip("Floor grid size (1.0x = adaptive default)")
        self._grid_size_slider.valueChanged.connect(self._on_grid_size_changed)
        self._grid_size_slider.sliderReleased.connect(self._on_grid_size_released)
        layout.addWidget(self._grid_size_slider)
        grid_meters = self._default_grid_size * self._grid_size_multiplier
        self._grid_size_label = QLabel(self._format_grid_label(self._grid_size_multiplier, grid_meters), self)
        self._grid_size_label.setFixedWidth(90)
        layout.addWidget(self._grid_size_label)

        layout.addSpacing(20)

        # Sphere size slider — same logarithmic mapping
        layout.addWidget(QLabel("Points:", self))
        self._sphere_size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._sphere_size_slider.setMinimum(0)
        self._sphere_size_slider.setMaximum(100)
        self._sphere_size_slider.setValue(self._multiplier_to_slider(self._sphere_size_multiplier))
        self._sphere_size_slider.setFixedWidth(100)
        self._sphere_size_slider.setToolTip("Point sphere size (1.0x = 12mm radius)")
        self._sphere_size_slider.valueChanged.connect(self._on_sphere_size_changed)
        self._sphere_size_slider.sliderReleased.connect(self._on_sphere_size_released)
        layout.addWidget(self._sphere_size_slider)
        sphere_mm = _DEFAULT_SPHERE_RADIUS * self._sphere_size_multiplier * 1000
        self._sphere_size_label = QLabel(self._format_sphere_label(self._sphere_size_multiplier, sphere_mm), self)
        self._sphere_size_label.setFixedWidth(90)
        layout.addWidget(self._sphere_size_label)

        layout.addStretch()

        return bar

    # -------------------------------------------------------------------------
    # Label formatting helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_grid_label(multiplier: float, meters: float) -> str:
        """Format grid size label with computed meters (or cm if < 1m)."""
        if meters < 1.0:
            return f"{multiplier:.1f}x ({meters * 100:.0f} cm)"
        return f"{multiplier:.1f}x ({meters:.1f} m)"

    @staticmethod
    def _format_sphere_label(multiplier: float, mm: float) -> str:
        """Format sphere size label with computed millimeters."""
        return f"{multiplier:.1f}x ({mm:.0f} mm)"

    # -------------------------------------------------------------------------
    # Slider value <-> multiplier conversion
    # -------------------------------------------------------------------------

    @staticmethod
    def _slider_to_multiplier(value: int) -> float:
        """Convert slider position [0, 100] to a log-scale multiplier.

        Position 50 = 1.0x (default). The mapping is 10^((value - 50) / 50),
        giving a range of 0.1x at position 0 to 10x at position 100.
        """
        return 10.0 ** ((value - 50) / 50.0)

    @staticmethod
    def _multiplier_to_slider(multiplier: float) -> int:
        """Convert a log-scale multiplier to slider position [0, 100].

        Inverse of _slider_to_multiplier. Position 50 = 1.0x.
        Clamps to valid slider range [0, 100].
        """
        return max(0, min(100, round(50 + 50 * math.log10(max(multiplier, 0.1)))))

    # -------------------------------------------------------------------------
    # Appearance slider callbacks
    # -------------------------------------------------------------------------

    def _on_cam_size_changed(self, value: int) -> None:
        """Update camera scale state and schedule a debounced geometry rebuild.

        Label updates are immediate (cheap). The actual geometry rebuild is
        deferred via a 50ms single-shot timer so that dragging the slider
        doesn't trigger 10-20 rebuilds — only one, after the user pauses.
        """
        if self._camera_scale_override is not None:
            return  # Explicit scale overrides slider

        self._camera_scale_multiplier = self._slider_to_multiplier(value)
        self._cam_size_label.setText(f"{self._camera_scale_multiplier:.1f}x")
        self._camera_scale = self._default_camera_scale * self._camera_scale_multiplier
        self._cam_rebuild_timer.start()  # Restart debounce timer

    def _on_grid_size_changed(self, value: int) -> None:
        """Update grid size state and schedule a debounced geometry rebuild."""
        self._grid_size_multiplier = self._slider_to_multiplier(value)
        grid_meters = self._default_grid_size * self._grid_size_multiplier
        self._grid_size_label.setText(self._format_grid_label(self._grid_size_multiplier, grid_meters))
        self._grid_rebuild_timer.start()  # Restart debounce timer

    def _on_sphere_size_changed(self, value: int) -> None:
        """Update sphere size immediately. setRadius() is a cheap batched property change."""
        self._sphere_size_multiplier = self._slider_to_multiplier(value)
        sphere_mm = _DEFAULT_SPHERE_RADIUS * self._sphere_size_multiplier * 1000
        self._sphere_size_label.setText(self._format_sphere_label(self._sphere_size_multiplier, sphere_mm))
        if self._sphere_cloud is not None:
            self._sphere_cloud.set_radius(_DEFAULT_SPHERE_RADIUS * self._sphere_size_multiplier)

    def _on_cam_size_released(self) -> None:
        """Emit camera size multiplier when user releases the slider."""
        self.camera_size_multiplier_changed.emit(self._camera_scale_multiplier)

    def _on_grid_size_released(self) -> None:
        """Emit grid size multiplier when user releases the slider."""
        self.grid_size_multiplier_changed.emit(self._grid_size_multiplier)

    def _on_sphere_size_released(self) -> None:
        """Emit sphere size multiplier when user releases the slider."""
        self.sphere_size_multiplier_changed.emit(self._sphere_size_multiplier)

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

        # Clear active references so _on_sync_index_changed returns early
        # during slider range updates below (stale geometry guard).
        # Dynamic entities are already retained via _retained_entities at creation time.
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

        # Rebuild scene content under the permanent root. _build_scene() retires
        # the old _scene container (disable + append to _retained_entities) and
        # creates a new one. The root entity is never replaced — setRootEntity()
        # is called exactly once in __init__.
        self._build_scene()

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
        """Show or hide the playback control bar (play/pause, speed, frame slider).

        The appearance controls (camera size, grid size) remain visible.
        Use this when embedding the widget in a container with shared controls.
        """
        self._playback_bar.setVisible(visible)

    def show_appearance_controls(self, visible: bool) -> None:
        """Show or hide the appearance control bar (camera size, grid size sliders)."""
        self._appearance_bar.setVisible(visible)

    def suspend_rendering(self) -> None:
        """Suspend Qt3D rendering by switching to OnDemand render policy.

        Stops the render thread's continuous loop. Critical during heavy
        background processing (reconstruction) because Mesa llvmpipe's
        software renderer can conflict with multi-threaded video decode
        when both compete for CPU/memory resources.
        """
        settings = self._view.renderSettings()
        if settings is not None:
            settings.setRenderPolicy(Qt3DRender.QRenderSettings.RenderPolicy.OnDemand)
            logger.info("Qt3D rendering suspended (OnDemand policy)")

    def resume_rendering(self) -> None:
        """Resume Qt3D rendering by switching back to Always render policy.

        Restores continuous rendering after background processing completes.
        """
        settings = self._view.renderSettings()
        if settings is not None:
            settings.setRenderPolicy(Qt3DRender.QRenderSettings.RenderPolicy.Always)
            logger.info("Qt3D rendering resumed (Always policy)")

    def capture_screenshot(self) -> QImage | None:
        """Capture the Qt3D scene via QRenderCapture.

        widget.grab() cannot capture Qt3D content because Qt3DWindow renders
        to its own GPU surface via createWindowContainer. QRenderCapture
        intercepts the render pipeline directly and returns the framebuffer
        contents as a QImage.

        QRenderCapture is created lazily on first call because attaching it
        to the frame graph crashes under xvfb/software rendering (Mesa
        llvmpipe segfault in glReadPixels path).

        Returns:
            QImage on success, None if the capture does not complete within
            500ms (e.g., render pipeline not yet initialised).
        """
        from PySide6.QtCore import QEventLoop

        if self._render_capture is None:
            self._render_capture = Qt3DRender.QRenderCapture(self._view.activeFrameGraph())

        reply = self._render_capture.requestCapture()

        loop = QEventLoop()
        reply.completed.connect(loop.quit)
        QTimer.singleShot(500, loop.quit)
        loop.exec()

        if reply.isComplete():
            return reply.image()
        return None
