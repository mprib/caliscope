"""
PyVista-based 3D visualization widget for triangulated motion capture playback.

Displays camera frustums and animated 3D point positions from WorldPoints data,
with playback controls for play/pause, looping, speed adjustment, and frame scrubbing.

Note: Inherits from QWidget (not QMainWindow) so it can be embedded in layouts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyvista as pv
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from pyvistaqt import QtInteractor

from caliscope import ICONS_DIR
from caliscope.ui.viz.playback_view_model import PlaybackViewModel

logger = logging.getLogger(__name__)


def _icon(name: str) -> QIcon:
    """Load SVG icon from gui/icons directory."""
    return QIcon(str(ICONS_DIR / f"{name}.svg"))


class PlaybackTriangulationWidgetPyVista(QWidget):
    """
    PyVista-based widget for animated playback of triangulated 3D points.

    Features:
    - Play/pause animation with configurable speed (0.1x to 3.0x)
    - Loop toggle for continuous playback
    - Frame slider for manual scrubbing
    - Camera label visibility toggle

    Note: Inherits from QWidget (not QMainWindow) so it can be embedded in layouts.
    """

    def __init__(self, view_model: PlaybackViewModel, parent: QWidget | None = None):
        super().__init__(parent)

        # Lazy import to ensure QApplication exists before pyvistaqt init
        from pyvistaqt import QtInteractor

        self.view_model = view_model
        self.sync_index: int = self.view_model.min_index

        # UI State
        self.show_camera_labels = True
        self.is_playing = False
        self.loop_enabled = True
        self.speed_multiplier = 1.0

        # Persistent Mesh References (The "Template" Pattern)
        self._point_cloud_mesh: pv.PolyData | None = None
        self._wireframe_mesh: pv.PolyData | None = None
        self._label_actor = None

        # Cache icons for play/pause toggle
        self._play_icon = _icon("play")
        self._pause_icon = _icon("pause")

        # Timer for animation
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 3D view (takes most space)
        self.plotter: QtInteractor = QtInteractor(parent=self)
        main_layout.addWidget(self.plotter, stretch=1)

        # Controls bar at bottom
        controls_widget = self._create_controls()
        main_layout.addWidget(controls_widget)

        # --- Initialization ---
        self._initialize_scene()
        self._create_static_actors()
        self._create_dynamic_actors()  # Create the persistent meshes once
        self._on_sync_index_changed(self.sync_index)  # Render first frame

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

        # Right side: labels toggle and frame slider
        self.labels_checkbox = QCheckBox("Camera Labels", self)
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self._on_labels_toggled)
        layout.addWidget(self.labels_checkbox)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(self.view_model.min_index)
        self.slider.setMaximum(self.view_model.max_index)
        self.slider.setValue(self.sync_index)
        self.slider.valueChanged.connect(self._on_sync_index_changed)
        layout.addWidget(self.slider, stretch=1)

        return controls

    def _initialize_scene(self):
        self.plotter.set_background("black")
        self.plotter.show_axes()
        self.plotter.camera_position = [(4, 4, 4), (0, 0, 0), (0, 0, 1)]
        self.plotter.show_grid()
        self.plotter.enable_trackball_style()
        logger.info("PyVista scene initialized")

    def _create_static_actors(self):
        """Create static camera geometry."""
        # Use smaller camera scale for playback (points are the focus, not cameras)
        camera_geom = self.view_model.get_camera_geometry(scale=0.0002)
        if camera_geom is None:
            return

        mesh = pv.PolyData(camera_geom["vertices"], faces=camera_geom["faces"])
        mesh.point_data["colors"] = camera_geom["colors"]

        self.plotter.add_mesh(
            mesh,
            name="camera_array",
            scalars="colors",
            rgb=True,
            opacity=0.7,
        )

        # Labels
        label_positions = [pos for pos, _ in camera_geom["labels"]]
        label_texts = [text for _, text in camera_geom["labels"]]

        self._label_actor = self.plotter.add_point_labels(
            label_positions,
            label_texts,
            font_size=12,
            point_color="white",
            text_color="white",
            point_size=1,
            name="camera_labels",
        )

    def _create_dynamic_actors(self):
        """
        Initialize the persistent PolyData objects for points and wireframes.
        This is called EXACTLY ONCE.
        """
        # Camera-only mode: no points to render
        if self.view_model.n_points == 0:
            logger.info("Camera-only mode: skipping dynamic actors (no points)")
            return

        # 1. Get initial frame data (likely NaN, but sets the size)
        frame_geom = self.view_model.get_frame_geometry(self.sync_index)

        # 2. Get static wireframe topology
        lines, line_colors = self.view_model.get_static_wireframe_data()

        # --- Point Cloud Mesh ---
        self._point_cloud_mesh = pv.PolyData(frame_geom.points)
        self._point_cloud_mesh.point_data["colors"] = frame_geom.colors

        self.plotter.add_mesh(
            self._point_cloud_mesh,
            name="mediapipe_points",
            render_points_as_spheres=True,
            point_size=10,
            scalars="colors",
            rgb=True,
            reset_camera=False,
        )

        # --- Wireframe Mesh ---
        # Note: We create a separate PolyData but it will be updated with the
        # same point coordinates in the loop.
        self._wireframe_mesh = pv.PolyData(frame_geom.points, lines=lines)

        # Wireframe colors are static (Cell Data), so we set them once here
        if len(line_colors) > 0:
            self._wireframe_mesh.cell_data["colors"] = line_colors

        # Only pass scalars/rgb when we have color data
        wireframe_kwargs: dict = {
            "name": "wireframe",
            "render_lines_as_tubes": True,
            "line_width": 5,
            "reset_camera": False,
        }
        if len(line_colors) > 0:
            wireframe_kwargs["scalars"] = "colors"
            wireframe_kwargs["rgb"] = True

        self.plotter.add_mesh(self._wireframe_mesh, **wireframe_kwargs)

        logger.info("Dynamic actors initialized with persistent mesh buffers")

    def _on_sync_index_changed(self, sync_index: int):
        """
        Update the coordinates of the existing meshes.
        Zero memory allocation, zero topology rebuilding.
        """
        self.sync_index = sync_index

        if self._point_cloud_mesh is None or self._wireframe_mesh is None:
            return

        # 1. Get the pre-scattered buffer (N, 3)
        frame_geom = self.view_model.get_frame_geometry(sync_index)

        # 2. Update Point Cloud
        # We update the numpy array in-place on the VTK object
        self._point_cloud_mesh.points = frame_geom.points
        self._point_cloud_mesh.point_data["colors"] = frame_geom.colors

        # 3. Update Wireframe
        # It shares the same vertices, so we just push the same buffer.
        # The 'lines' topology we set in __init__ remains valid.
        # Points that are NaN will simply not render lines connected to them.
        self._wireframe_mesh.points = frame_geom.points

        # 4. Render
        self.plotter.render()

    def _toggle_playback(self, checked: bool):
        self.is_playing = checked
        if checked:
            self.play_button.setIcon(self._pause_icon)
            self._start_playback()
        else:
            self.play_button.setIcon(self._play_icon)
            self.playback_timer.stop()

    def _start_playback(self):
        if self.view_model.frame_rate <= 0:
            return
        interval_ms = int(1000 / (self.view_model.frame_rate * self.speed_multiplier))
        self.playback_timer.start(interval_ms)

    def _advance_frame(self):
        next_index = self.sync_index + 1
        if next_index > self.view_model.max_index:
            if self.loop_enabled:
                next_index = self.view_model.min_index
            else:
                self.play_button.setChecked(False)
                self.playback_timer.stop()
                return
        self.slider.setValue(next_index)

    def _toggle_loop(self, checked: bool):
        self.loop_enabled = checked
        self.loop_button.setChecked(checked)

    def _on_speed_changed(self, value: int):
        self.speed_multiplier = value / 10.0
        self.speed_label.setText(f"{self.speed_multiplier:.1f}x")
        if self.is_playing:
            self._start_playback()

    def _on_labels_toggled(self, state: int):
        self.show_camera_labels = state == Qt.CheckState.Checked.value
        if hasattr(self, "_label_actor") and self._label_actor:
            self._label_actor.SetVisibility(self.show_camera_labels)
            self.plotter.render()

    def showEvent(self, event) -> None:
        """Force render when widget becomes visible."""
        super().showEvent(event)
        self.plotter.render()

    def set_view_model(self, view_model: PlaybackViewModel) -> None:
        """
        Replace the ViewModel and rebuild the scene.

        Used when switching recordings or trackers in post-processing.
        Must rebuild the full scene (not just dynamic actors) because
        cameras may differ between recordings.
        """
        logger.info(f"Switching view model: {self.view_model.n_points} points → {view_model.n_points} points")

        # Stop any playback
        self.playback_timer.stop()
        self.is_playing = False
        self.play_button.setChecked(False)
        self.play_button.setIcon(self._play_icon)

        # Clear mesh references FIRST - before any slider changes that might
        # trigger _on_sync_index_changed() with mismatched point counts
        self._point_cloud_mesh = None
        self._wireframe_mesh = None

        self.view_model = view_model
        self.sync_index = view_model.min_index

        # Update slider range (setValue may trigger _on_sync_index_changed,
        # but mesh refs are already None so it safely returns early)
        self.slider.setMinimum(view_model.min_index)
        self.slider.setMaximum(view_model.max_index)
        self.slider.setValue(self.sync_index)

        # FULL scene rebuild
        self.plotter.clear()
        self._initialize_scene()
        self._create_static_actors()
        self._create_dynamic_actors()
        self._on_sync_index_changed(self.sync_index)
