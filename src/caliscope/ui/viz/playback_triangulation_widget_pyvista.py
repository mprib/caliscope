"""
PyVista-based 3D visualization widget for triangulated motion capture playback.

Displays camera frustums and animated 3D point positions from WorldPoints data,
with playback controls for play/pause, looping, speed adjustment, and frame scrubbing.

Note: Inherits from QWidget (not QMainWindow) so it can be embedded in layouts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
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

    # Default camera frustum scale for production scenes (meters, small scale)
    DEFAULT_CAMERA_SCALE = 0.0002

    def __init__(
        self,
        view_model: PlaybackViewModel,
        parent: QWidget | None = None,
        camera_scale: float | None = None,
    ):
        super().__init__(parent)

        # Lazy import to ensure QApplication exists before pyvistaqt init
        from pyvistaqt import QtInteractor

        self.view_model = view_model
        self._camera_scale = camera_scale if camera_scale is not None else self.DEFAULT_CAMERA_SCALE
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

        # Reduce CPU by tuning VTK interactor timing
        # (especially important when using software rendering like llvmpipe)
        if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
            vtk_iren = self.plotter.iren.interactor
            if vtk_iren is not None:
                # Event polling: 33ms = 30Hz (default 10ms = 100Hz is overkill)
                vtk_iren.SetTimerDuration(33)
                # Render rate limits
                vtk_iren.SetDesiredUpdateRate(15.0)  # Max 15 FPS during rotate/zoom
                vtk_iren.SetStillUpdateRate(0.5)  # 0.5 FPS when idle

        # Controls bar at bottom
        self._control_bar = self._create_controls()
        main_layout.addWidget(self._control_bar)

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
        # Log GPU capabilities for debugging CPU issues
        if hasattr(self.plotter, "ren_win"):
            caps = self.plotter.ren_win.ReportCapabilities()
            # First line contains renderer name (e.g., "OpenGL renderer: llvmpipe" or "NVIDIA...")
            first_line = caps.split("\n")[0] if caps else "unknown"
            logger.info(f"VTK Renderer: {first_line}")

        self.plotter.set_background("black")
        self.plotter.show_axes()
        self._set_adaptive_camera()
        self.plotter.show_grid()
        # Terrain style keeps Z-up (vertical stays vertical) during rotation
        self.plotter.enable_terrain_style(mouse_wheel_zooms=True)
        logger.info("PyVista scene initialized")

    def _add_origin_axes(self) -> None:
        """Add small XYZ axes arrows at the origin (0,0,0).

        Uses standard colors: X=red, Y=green, Z=blue.
        Arrow size is scaled based on camera positions.
        """
        positions = self.view_model.get_camera_positions()
        if positions is None or len(positions) == 0:
            arrow_length = 0.1
        else:
            # Scale arrows to ~5% of scene extent
            extent = positions.max(axis=0) - positions.min(axis=0)
            arrow_length = max(extent.max() * 0.05, 0.02)

        # X-axis (red)
        x_arrow = pv.Arrow(
            start=(0, 0, 0),
            direction=(1, 0, 0),
            scale=arrow_length,
            tip_length=0.3,
            tip_radius=0.15,
            shaft_radius=0.05,
        )
        self.plotter.add_mesh(x_arrow, name="origin_x", color="red", opacity=0.9)

        # Y-axis (green)
        y_arrow = pv.Arrow(
            start=(0, 0, 0),
            direction=(0, 1, 0),
            scale=arrow_length,
            tip_length=0.3,
            tip_radius=0.15,
            shaft_radius=0.05,
        )
        self.plotter.add_mesh(y_arrow, name="origin_y", color="green", opacity=0.9)

        # Z-axis (blue)
        z_arrow = pv.Arrow(
            start=(0, 0, 0),
            direction=(0, 0, 1),
            scale=arrow_length,
            tip_length=0.3,
            tip_radius=0.15,
            shaft_radius=0.05,
        )
        self.plotter.add_mesh(z_arrow, name="origin_z", color="blue", opacity=0.9)

    def _add_floor_indicator(self) -> None:
        """Add a semi-transparent floor plane at z=0 centered on the origin.

        The floor is always centered at (0,0,0) - the world origin.
        Size is based on camera positions to ensure it's appropriately scaled.
        """
        positions = self.view_model.get_camera_positions()
        if positions is None or len(positions) == 0:
            floor_size = 2.0
        else:
            # Size floor to encompass cameras plus margin
            # Use max distance from origin to any camera
            distances = np.sqrt((positions[:, :2] ** 2).sum(axis=1))
            floor_size = max(distances.max() * 2.5, 0.5)

        # Create plane centered at origin (0, 0, 0)
        floor = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=floor_size,
            j_size=floor_size,
            i_resolution=10,
            j_resolution=10,
        )

        self.plotter.add_mesh(
            floor,
            name="floor",
            color="#333333",
            opacity=0.3,
            show_edges=True,
            edge_color="#555555",
            lighting=False,
        )

    def _set_adaptive_camera(self) -> None:
        """Set camera position based on scene extent.

        For meter-scale real scenes (~1-2m), uses default view.
        For mm-scale synthetic scenes (~2000-4000mm), positions camera
        to show the full capture volume.
        """
        # Get scene extent from camera positions
        positions = self.view_model.get_camera_positions()
        if positions is None or len(positions) == 0:
            # Fallback: default position for unknown scenes
            self.plotter.camera_position = [(4, 4, 4), (0, 0, 0), (0, 0, 1)]
            return

        # Compute bounding box of camera positions
        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)
        center = (min_coords + max_coords) / 2
        extent = max_coords - min_coords
        max_extent = max(extent)

        # Position camera at ~2x max extent, 45° angle
        # This ensures the full scene is visible with comfortable margins
        dist = max_extent * 2.0
        eye = center + [dist * 0.7, dist * 0.7, dist * 0.5]

        self.plotter.camera_position = [
            tuple(eye),
            tuple(center),
            (0, 0, 1),  # Z-up
        ]

    def _create_static_actors(self):
        """Create static camera geometry and floor indicator."""
        camera_geom = self.view_model.get_camera_geometry(scale=self._camera_scale)
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

        # Floor plane at z=0 to indicate origin/ground level
        self._add_floor_indicator()

        # Small XYZ axes at the true origin
        self._add_origin_axes()

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

    def suspend_vtk(self) -> None:
        """Pause VTK interactor to reduce idle CPU when widget is not active.

        VTK's RenderWindowInteractor runs a repeating timer (~10ms) that
        continuously polls for events. When the tab is switched away,
        this timer keeps running, wasting CPU. Disabling the interactor
        while inactive eliminates this overhead.

        Call this when the containing tab becomes inactive.
        Pair with resume_vtk() when tab becomes active again.
        """
        logger.debug("PlaybackTriangulationWidgetPyVista suspending VTK")
        if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
            # Access underlying VTK interactor - PyVista wrapper doesn't expose these methods
            vtk_iren = self.plotter.iren.interactor
            if vtk_iren is not None:
                # Disable() stops the event processing timer entirely (not just rendering)
                vtk_iren.Disable()

    def resume_vtk(self) -> None:
        """Resume VTK interactor when widget becomes active.

        Re-enables rendering and triggers an immediate render to refresh
        the view after suspension.

        Call this when the containing tab becomes active.
        """
        logger.debug("PlaybackTriangulationWidgetPyVista resuming VTK")
        if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
            vtk_iren = self.plotter.iren.interactor
            if vtk_iren is not None:
                # Enable() resumes event processing
                vtk_iren.Enable()
        self.plotter.render()

    def showEvent(self, event) -> None:
        """Resume interactor when widget becomes visible (actual visibility change)."""
        super().showEvent(event)
        self.resume_vtk()

    def hideEvent(self, event) -> None:
        """Pause interactor when widget is hidden (actual visibility change)."""
        super().hideEvent(event)
        self.suspend_vtk()

    def set_view_model(self, view_model: PlaybackViewModel, preserve_camera: bool = False) -> None:
        """
        Replace the ViewModel and rebuild the scene.

        Used when switching recordings or trackers in post-processing.
        Must rebuild the full scene (not just dynamic actors) because
        cameras may differ between recordings.

        Args:
            view_model: New view model to display
            preserve_camera: If True, restore camera position and sync_index
                after rebuild (useful for coordinate transforms)
        """
        logger.info(f"Switching view model: {self.view_model.n_points} points → {view_model.n_points} points")

        # Save state if preserving
        saved_camera = self.plotter.camera_position if preserve_camera else None
        saved_sync_index = self.sync_index if preserve_camera else None

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

        # Determine which sync_index to use
        if saved_sync_index is not None:
            # Clamp to valid range in new view model
            self.sync_index = max(view_model.min_index, min(saved_sync_index, view_model.max_index))
        else:
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

        # Restore camera if preserving
        if saved_camera is not None:
            self.plotter.camera_position = saved_camera
            self.plotter.render()

    def set_sync_index(self, sync_index: int) -> None:
        """
        Set frame index programmatically (for external slider control).

        Use this when embedding the widget in a container with a shared slider.
        """
        self.sync_index = sync_index
        self._on_sync_index_changed(sync_index)

        # Update internal slider without triggering signal
        self.slider.blockSignals(True)
        self.slider.setValue(sync_index)
        self.slider.blockSignals(False)

    def show_playback_controls(self, visible: bool) -> None:
        """
        Show or hide the playback control bar.

        Use this when embedding the widget in a container with shared controls.

        Args:
            visible: If False, hides the slider, play/pause button, speed control, etc.
        """
        self._control_bar.setVisible(visible)
