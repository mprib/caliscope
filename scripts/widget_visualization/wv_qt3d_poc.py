"""
Qt3D Proof-of-Concept: replicate all PyVista primitives used in PlaybackVizWidget.

Demonstrates:
1. Camera frustums (pyramid meshes with per-vertex color)
2. Animated point cloud via INSTANCED SPHERES (QPointSize unsupported under RHI)
3. Wireframe lines (fixed topology, positions updated per-frame)
4. Floor plane (semi-transparent grid)
5. Origin axes (colored arrows)
6. Orbit camera controller
7. Play/pause with manual slider scrubbing

Uses synthetic data — no real calibration needed.
Run: uv run python scripts/widget_visualization/wv_qt3d_poc.py
"""

import math
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QMouseEvent, QVector3D, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
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


# ---------------------------------------------------------------------------
# Terrain-style camera controller (Z-up locked)
# ---------------------------------------------------------------------------


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

    def _apply(self):
        """Recompute camera position from spherical coords and apply."""
        # Clamp elevation to avoid gimbal lock and upside-down
        self._elevation = max(math.radians(5), min(math.radians(85), self._elevation))
        self._distance = max(0.1, self._distance)

        # Spherical to cartesian (Z-up)
        cos_el = math.cos(self._elevation)
        eye = QVector3D(
            self._distance * cos_el * math.cos(self._azimuth),
            self._distance * cos_el * math.sin(self._azimuth),
            self._distance * math.sin(self._elevation),
        )

        self._camera.setPosition(self._focus + eye)
        self._camera.setViewCenter(self._focus)
        self._camera.setUpVector(QVector3D(0, 0, 1))

    def mouse_press(self, event: QMouseEvent):
        self._last_pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_orbit = True
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._dragging_pan = True

    def mouse_release(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_orbit = False
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._dragging_pan = False

    def mouse_move(self, event: QMouseEvent):
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos

        if self._dragging_orbit:
            self._azimuth -= math.radians(dx * self._orbit_speed)
            self._elevation += math.radians(dy * self._orbit_speed)
            self._apply()

        elif self._dragging_pan:
            # Pan in screen-aligned plane
            # Right vector (horizontal in screen space, Z-up world)
            right = QVector3D(
                -math.sin(self._azimuth),
                math.cos(self._azimuth),
                0,
            )
            # Up vector projected into screen (not world Z, but screen vertical)
            # This is the component of world-Z perpendicular to the view direction
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

    def wheel(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.0 - delta * self._zoom_speed
        self._distance *= factor
        self._apply()

    def set_focus(self, focus: QVector3D):
        """Programmatically set the focus point."""
        self._focus = focus
        self._apply()

    def set_distance(self, distance: float):
        """Programmatically set the orbit distance."""
        self._distance = distance
        self._apply()


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from utils import capture_widget, clear_output_dir, process_events_for  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def make_synthetic_cameras(n_cams: int = 4, radius: float = 2.0):
    """Generate camera positions in a ring, looking at origin."""
    angles = np.linspace(0, 2 * np.pi, n_cams, endpoint=False)
    cameras = []
    for i, a in enumerate(angles):
        pos = np.array([radius * np.cos(a), radius * np.sin(a), 1.0], dtype=np.float32)
        forward = -pos / np.linalg.norm(pos)
        up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        scale = 0.15
        half_w, half_h = 0.12 * scale, 0.08 * scale
        tip = pos
        center = pos + forward * scale
        corners = [
            center + right * half_w + up * half_h,
            center - right * half_w + up * half_h,
            center - right * half_w - up * half_h,
            center + right * half_w - up * half_h,
        ]
        cameras.append(
            {
                "id": i,
                "apex": tip,
                "corners": np.array(corners, dtype=np.float32),
            }
        )
    return cameras


def make_synthetic_trajectory(n_frames: int = 120, n_points: int = 10):
    """Generate points moving in a figure-8 pattern."""
    t = np.linspace(0, 2 * np.pi, n_frames, dtype=np.float32)
    trajectories = np.zeros((n_frames, n_points, 3), dtype=np.float32)

    for p in range(n_points):
        phase = p * 2 * np.pi / n_points
        spread = 0.3 + 0.1 * p
        trajectories[:, p, 0] = spread * np.sin(t + phase)
        trajectories[:, p, 1] = spread * np.sin(2 * t + phase) * 0.5
        trajectories[:, p, 2] = 0.5 + 0.2 * np.sin(t * 3 + phase)

    return trajectories


WIREFRAME_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 0),
]


# ---------------------------------------------------------------------------
# Qt3D helpers
# ---------------------------------------------------------------------------


def numpy_to_qbytearray(arr: np.ndarray) -> QByteArray:
    """Convert numpy array to QByteArray for Qt3D buffers."""
    return QByteArray(arr.astype(np.float32).tobytes())


def create_custom_mesh_entity(
    vertices: np.ndarray,
    indices: np.ndarray,
    color: QColor,
    parent: Qt3DCore.QEntity,
    opacity: float = 1.0,
) -> Qt3DCore.QEntity:
    """Create a triangle mesh entity from raw vertex/index arrays."""
    entity = Qt3DCore.QEntity(parent)

    geometry = Qt3DCore.QGeometry(entity)

    # Vertex buffer
    vertex_buf = Qt3DCore.QBuffer(geometry)
    vertex_buf.setData(numpy_to_qbytearray(vertices))

    pos_attr = Qt3DCore.QAttribute(geometry)
    pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
    pos_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
    pos_attr.setVertexSize(3)
    pos_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.VertexAttribute)
    pos_attr.setBuffer(vertex_buf)
    pos_attr.setByteStride(3 * 4)
    pos_attr.setCount(len(vertices))
    geometry.addAttribute(pos_attr)

    # Index buffer
    index_data = indices.astype(np.uint32)
    index_buf = Qt3DCore.QBuffer(geometry)
    index_buf.setData(QByteArray(index_data.tobytes()))

    index_attr = Qt3DCore.QAttribute(geometry)
    index_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.UnsignedInt)
    index_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.IndexAttribute)
    index_attr.setBuffer(index_buf)
    index_attr.setCount(len(index_data))
    geometry.addAttribute(index_attr)

    renderer = Qt3DRender.QGeometryRenderer(entity)
    renderer.setGeometry(geometry)
    renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.PrimitiveType.Triangles)

    material = Qt3DExtras.QPhongAlphaMaterial(entity)
    material.setAmbient(color)
    material.setDiffuse(color)
    material.setAlpha(opacity)

    entity.addComponent(renderer)
    entity.addComponent(material)

    return entity


def create_line_entity(
    vertices: np.ndarray,
    indices: np.ndarray,
    color: QColor,
    parent: Qt3DCore.QEntity,
) -> tuple[Qt3DCore.QEntity, Qt3DCore.QBuffer]:
    """Create a line entity, returning (entity, vertex_buffer) for later updates."""
    entity = Qt3DCore.QEntity(parent)

    geometry = Qt3DCore.QGeometry(entity)

    vertex_buf = Qt3DCore.QBuffer(geometry)
    vertex_buf.setData(numpy_to_qbytearray(vertices))

    pos_attr = Qt3DCore.QAttribute(geometry)
    pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
    pos_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
    pos_attr.setVertexSize(3)
    pos_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.VertexAttribute)
    pos_attr.setBuffer(vertex_buf)
    pos_attr.setByteStride(3 * 4)
    pos_attr.setCount(len(vertices))
    geometry.addAttribute(pos_attr)

    index_data = indices.astype(np.uint32)
    index_buf = Qt3DCore.QBuffer(geometry)
    index_buf.setData(QByteArray(index_data.tobytes()))

    index_attr = Qt3DCore.QAttribute(geometry)
    index_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.UnsignedInt)
    index_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.IndexAttribute)
    index_attr.setBuffer(index_buf)
    index_attr.setCount(len(index_data))
    geometry.addAttribute(index_attr)

    renderer = Qt3DRender.QGeometryRenderer(entity)
    renderer.setGeometry(geometry)
    renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.PrimitiveType.Lines)

    material = Qt3DExtras.QPhongMaterial(entity)
    material.setAmbient(color)
    material.setDiffuse(color)

    entity.addComponent(renderer)
    entity.addComponent(material)

    return entity, vertex_buf


class SphereCloud:
    """A collection of sphere entities that can be repositioned per-frame.

    Each point gets its own QEntity with a QSphereMesh and QTransform.
    For motion capture landmark counts (10-50 points), individual entities
    are simpler and more reliable than instanced rendering, which requires
    custom geometry setup that Qt3D's Python bindings make awkward.

    To update positions: call update_positions(new_positions).
    To show/hide: call set_enabled(bool).
    """

    def __init__(
        self,
        positions: np.ndarray,
        color: QColor,
        parent: Qt3DCore.QEntity,
        sphere_radius: float = 0.02,
    ):
        self._parent_entity = Qt3DCore.QEntity(parent)
        self._transforms: list[Qt3DCore.QTransform] = []
        self._entities: list[Qt3DCore.QEntity] = []
        n_points = len(positions)

        # Shared material — all spheres get the same color
        # (Qt3D allows sharing components across entities)
        shared_mesh = Qt3DExtras.QSphereMesh()
        shared_mesh.setRadius(sphere_radius)
        shared_mesh.setRings(8)
        shared_mesh.setSlices(8)

        for i in range(n_points):
            entity = Qt3DCore.QEntity(self._parent_entity)

            # Each entity gets its own mesh instance (Qt3D requires 1:1 component-entity)
            mesh = Qt3DExtras.QSphereMesh(entity)
            mesh.setRadius(sphere_radius)
            mesh.setRings(8)
            mesh.setSlices(8)

            material = Qt3DExtras.QPhongMaterial(entity)
            material.setAmbient(color)
            material.setDiffuse(color)

            transform = Qt3DCore.QTransform(entity)
            pos = positions[i]
            transform.setTranslation(QVector3D(float(pos[0]), float(pos[1]), float(pos[2])))

            entity.addComponent(mesh)
            entity.addComponent(material)
            entity.addComponent(transform)

            self._transforms.append(transform)
            self._entities.append(entity)

    def update_positions(self, positions: np.ndarray) -> None:
        """Update sphere positions. NaN positions hide the sphere."""
        for i, transform in enumerate(self._transforms):
            pos = positions[i]
            if np.isnan(pos[0]):
                # Move off-screen for NaN (Qt3D has no per-entity visibility toggle
                # without removing components, but moving far away is cheap)
                transform.setTranslation(QVector3D(99999, 99999, 99999))
            else:
                transform.setTranslation(QVector3D(float(pos[0]), float(pos[1]), float(pos[2])))

    def set_enabled(self, enabled: bool) -> None:
        """Show or hide the entire point cloud."""
        self._parent_entity.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------


def create_double_sided_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
    color: QColor,
    parent: Qt3DCore.QEntity,
) -> Qt3DCore.QEntity:
    """Create a mesh that renders from both sides by duplicating triangles.

    Uses opaque QPhongMaterial to avoid alpha depth-sorting issues.
    """
    # Duplicate triangles with reversed winding for double-sided rendering
    reversed_indices = indices.reshape(-1, 3)[:, ::-1].flatten()
    all_indices = np.concatenate([indices, reversed_indices]).astype(np.uint32)

    entity = Qt3DCore.QEntity(parent)

    geometry = Qt3DCore.QGeometry(entity)

    vertex_buf = Qt3DCore.QBuffer(geometry)
    vertex_buf.setData(numpy_to_qbytearray(vertices))

    pos_attr = Qt3DCore.QAttribute(geometry)
    pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
    pos_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
    pos_attr.setVertexSize(3)
    pos_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.VertexAttribute)
    pos_attr.setBuffer(vertex_buf)
    pos_attr.setByteStride(3 * 4)
    pos_attr.setCount(len(vertices))
    geometry.addAttribute(pos_attr)

    index_buf = Qt3DCore.QBuffer(geometry)
    index_buf.setData(QByteArray(all_indices.tobytes()))

    index_attr = Qt3DCore.QAttribute(geometry)
    index_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.UnsignedInt)
    index_attr.setAttributeType(Qt3DCore.QAttribute.AttributeType.IndexAttribute)
    index_attr.setBuffer(index_buf)
    index_attr.setCount(len(all_indices))
    geometry.addAttribute(index_attr)

    renderer = Qt3DRender.QGeometryRenderer(entity)
    renderer.setGeometry(geometry)
    renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.PrimitiveType.Triangles)

    material = Qt3DExtras.QPhongAlphaMaterial(entity)
    material.setAmbient(color)
    material.setDiffuse(color)
    material.setAlpha(0.6)

    entity.addComponent(renderer)
    entity.addComponent(material)

    return entity


def build_camera_frustums(cameras: list[dict], parent: Qt3DCore.QEntity) -> list[dict]:
    """Add camera frustums with solid faces, highlighted edges, and label anchors."""
    label_anchors = []
    for cam in cameras:
        apex = cam["apex"]
        corners = cam["corners"]
        cam_id = cam["id"]

        vertices = np.vstack([apex.reshape(1, 3), corners]).astype(np.float32)

        # Solid faces (dark green, opaque, double-sided)
        tri_indices = np.array(
            [
                0,
                1,
                2,
                0,
                2,
                3,
                0,
                3,
                4,
                0,
                4,
                1,
                1,
                2,
                3,
                1,
                3,
                4,
            ],
            dtype=np.uint32,
        )

        create_double_sided_mesh(
            vertices,
            tri_indices,
            QColor(30, 120, 30),  # Darker green fill
            parent,
        )

        # Bright edge wireframe on top
        edge_indices = np.array(
            [
                0,
                1,
                0,
                2,
                0,
                3,
                0,
                4,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                1,
            ],
            dtype=np.uint32,
        )

        create_line_entity(
            vertices,
            edge_indices,
            QColor(80, 255, 80),  # Bright green edges
            parent,
        )

        label_anchors.append(
            {
                "cam_id": cam_id,
                "position": apex + np.array([0, 0, 0.05], dtype=np.float32),
            }
        )

    return label_anchors


def build_floor_grid(
    parent: Qt3DCore.QEntity,
    size: float = 5.0,
    spacing: float = 0.5,
    color: QColor = QColor(80, 80, 80),
    axis_color: QColor = QColor(120, 120, 120),
) -> None:
    """Add a grid at z=0 for spatial reference.

    Draws lines along X and Y at regular intervals. Lines through the
    origin are brighter to mark the principal axes.
    """
    half = size / 2
    n_lines = int(size / spacing)

    # Build grid line vertices and indices
    # We'll draw regular lines first, then overlay axis lines on top

    # --- Regular grid lines ---
    idx = 0
    regular_verts = []
    regular_indices = []

    for i in range(n_lines + 1):
        coord = -half + i * spacing
        if abs(coord) < spacing * 0.01:
            continue  # Skip origin lines — drawn separately as brighter

        # Line parallel to X axis (at this Y)
        regular_verts.extend(
            [
                [-half, coord, 0],
                [half, coord, 0],
            ]
        )
        regular_indices.extend([idx, idx + 1])
        idx += 2

        # Line parallel to Y axis (at this X)
        regular_verts.extend(
            [
                [coord, -half, 0],
                [coord, half, 0],
            ]
        )
        regular_indices.extend([idx, idx + 1])
        idx += 2

    if regular_verts:
        verts_arr = np.array(regular_verts, dtype=np.float32)
        idx_arr = np.array(regular_indices, dtype=np.uint32)
        create_line_entity(verts_arr, idx_arr, color, parent)

    # --- Origin axis lines (brighter) ---
    axis_verts = np.array(
        [
            [-half, 0, 0],
            [half, 0, 0],  # X axis line
            [0, -half, 0],
            [0, half, 0],  # Y axis line
        ],
        dtype=np.float32,
    )
    axis_indices = np.array([0, 1, 2, 3], dtype=np.uint32)
    create_line_entity(axis_verts, axis_indices, axis_color, parent)


def build_origin_axes(parent: Qt3DCore.QEntity, length: float = 0.3) -> None:
    """Add XYZ axis indicators at the origin."""
    axes = [
        (QVector3D(1, 0, 0), QColor("red"), QVector3D(0, 0, -90)),  # X
        (QVector3D(0, 1, 0), QColor("green"), QVector3D(0, 0, 0)),  # Y
        (QVector3D(0, 0, 1), QColor("blue"), QVector3D(90, 0, 0)),  # Z
    ]

    for direction, color, euler_rot in axes:
        entity = Qt3DCore.QEntity(parent)

        mesh = Qt3DExtras.QCylinderMesh(entity)
        mesh.setRadius(0.008)
        mesh.setLength(length)

        material = Qt3DExtras.QPhongMaterial(entity)
        material.setAmbient(color)
        material.setDiffuse(color)

        transform = Qt3DCore.QTransform(entity)
        offset = direction * length / 2
        transform.setTranslation(offset)
        transform.setRotationX(euler_rot.x())
        transform.setRotationZ(euler_rot.z())

        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class Qt3DPocWidget(QWidget):
    """Self-contained Qt3D proof of concept widget with playback controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Qt3D PoC — Caliscope 3D Primitives")

        # Generate synthetic data
        self._cameras = make_synthetic_cameras()
        self._trajectory = make_synthetic_trajectory()
        self._n_frames = len(self._trajectory)
        self._current_frame = 0
        self._is_playing = False

        # Build UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Qt3D window
        self._view = Qt3DExtras.Qt3DWindow()
        self._view.defaultFrameGraph().setClearColor(QColor(25, 25, 25))

        # Install event filter on the Qt3DWindow to capture mouse events
        # (createWindowContainer forwards events to the wrapped QWindow)
        self._view.installEventFilter(self)

        container = QWidget.createWindowContainer(self._view, self)
        layout.addWidget(container, stretch=1)

        # Controls bar
        controls = self._build_controls()
        layout.addWidget(controls)

        # Scene setup
        self._root = Qt3DCore.QEntity()
        self._setup_camera()
        self._build_scene()
        self._view.setRootEntity(self._root)

        # Animation timer (stopped by default — user controls playback)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

    def _build_controls(self) -> QWidget:
        controls = QWidget(self)
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(8, 4, 8, 4)

        # Play/Pause
        self._play_btn = QPushButton("Play", self)
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(60)
        self._play_btn.clicked.connect(self._toggle_playback)
        layout.addWidget(self._play_btn)

        # Speed control
        layout.addWidget(QLabel("Speed:", self))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._speed_slider.setMinimum(1)
        self._speed_slider.setMaximum(30)
        self._speed_slider.setValue(10)
        self._speed_slider.setFixedWidth(80)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_slider)
        self._speed_label = QLabel("1.0x", self)
        self._speed_label.setFixedWidth(35)
        layout.addWidget(self._speed_label)

        layout.addStretch()

        # Wireframe toggle
        self._wire_check = QCheckBox("Wireframe", self)
        self._wire_check.setChecked(True)
        self._wire_check.stateChanged.connect(self._on_wireframe_toggled)
        layout.addWidget(self._wire_check)

        # Points toggle
        self._points_check = QCheckBox("Points", self)
        self._points_check.setChecked(True)
        self._points_check.stateChanged.connect(self._on_points_toggled)
        layout.addWidget(self._points_check)

        layout.addStretch()

        # Frame slider
        self._frame_label = QLabel("Frame: 0", self)
        self._frame_label.setFixedWidth(70)
        layout.addWidget(self._frame_label)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimum(0)
        self._slider.setMaximum(self._n_frames - 1)
        self._slider.valueChanged.connect(self._on_frame_changed)
        layout.addWidget(self._slider, stretch=1)

        return controls

    def _setup_camera(self):
        camera = self._view.camera()
        camera.lens().setPerspectiveProjection(45.0, 16.0 / 9.0, 0.01, 100.0)

        # Terrain-style controller — Z stays up, no roll
        self._cam_controller = TerrainCameraController(camera)

    def eventFilter(self, obj, event):
        """Forward mouse events from Qt3DWindow to our terrain camera controller."""
        from PySide6.QtCore import QEvent

        if obj is self._view:
            if event.type() == QEvent.Type.MouseButtonPress:
                self._cam_controller.mouse_press(event)
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self._cam_controller.mouse_release(event)
                return True
            elif event.type() == QEvent.Type.MouseMove:
                self._cam_controller.mouse_move(event)
                return True
            elif event.type() == QEvent.Type.Wheel:
                self._cam_controller.wheel(event)
                return True
        return super().eventFilter(obj, event)

    def _build_scene(self):
        # Static geometry
        self._label_anchors = build_camera_frustums(self._cameras, self._root)
        build_floor_grid(self._root)
        build_origin_axes(self._root)

        # Sphere cloud for point visualization
        initial_points = self._trajectory[0]
        self._sphere_cloud = SphereCloud(
            initial_points,
            QColor(200, 200, 200),
            self._root,
            sphere_radius=0.02,
        )

        # Wireframe lines
        wire_indices = np.array(
            [[a, b] for a, b in WIREFRAME_PAIRS],
            dtype=np.uint32,
        ).flatten()
        self._wire_entity, self._wire_buffer = create_line_entity(
            initial_points,
            wire_indices,
            QColor(100, 180, 255),
            self._root,
        )

    # --- Playback controls ---

    def _toggle_playback(self, checked: bool):
        self._is_playing = checked
        self._play_btn.setText("Pause" if checked else "Play")
        if checked:
            speed = self._speed_slider.value() / 10.0
            interval_ms = max(1, int(33 / speed))
            self._timer.start(interval_ms)
        else:
            self._timer.stop()

    def _on_speed_changed(self, value: int):
        speed = value / 10.0
        self._speed_label.setText(f"{speed:.1f}x")
        if self._is_playing:
            interval_ms = max(1, int(33 / speed))
            self._timer.start(interval_ms)

    def _advance_frame(self):
        next_frame = (self._current_frame + 1) % self._n_frames
        self._slider.setValue(next_frame)

    def _on_frame_changed(self, frame: int):
        self._current_frame = frame
        self._frame_label.setText(f"Frame: {frame}")
        self._update_dynamic_geometry()

    def _on_wireframe_toggled(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self._wire_entity.setEnabled(enabled)

    def _on_points_toggled(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self._sphere_cloud.set_enabled(enabled)

    def _update_dynamic_geometry(self):
        """Push new point positions to the GPU buffers."""
        points = self._trajectory[self._current_frame]
        self._sphere_cloud.update_positions(points)
        self._wire_buffer.setData(numpy_to_qbytearray(points))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    clear_output_dir()
    app = QApplication(sys.argv)

    widget = Qt3DPocWidget()
    widget.resize(1200, 800)
    widget.show()

    # Give Qt3D time to initialize
    process_events_for(2000)

    # Capture screenshots for headless review
    capture_widget(widget, "01_qt3d_poc_initial.png")

    widget._slider.setValue(30)
    process_events_for(500)
    capture_widget(widget, "02_qt3d_poc_frame30.png")

    widget.resize(900, 600)
    process_events_for(500)
    capture_widget(widget, "03_qt3d_poc_small.png")

    print("\nScreenshots saved to scripts/widget_visualization/output/")

    if "--headless" in sys.argv:
        print("Headless mode — exiting after screenshots.")
        QTimer.singleShot(100, app.quit)
    else:
        print("Interactive mode — use slider to scrub, Play/Pause to animate.")
        print("Camera: left-drag to orbit (Z-up locked), right-drag to pan, scroll to zoom.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
