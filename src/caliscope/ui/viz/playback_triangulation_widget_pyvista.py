import logging
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMainWindow, QSlider, QCheckBox, QPushButton, QWidget, QHBoxLayout, QLabel
from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np

from caliscope.ui.viz.playback_view_model import PlaybackViewModel

logger = logging.getLogger(__name__)


class PlaybackTriangulationWidgetPyVista(QMainWindow):
    def __init__(self, view_model: PlaybackViewModel, parent=None):
        super().__init__(parent)
        self.view_model = view_model
        self.sync_index: int = self.view_model.min_index

        # UI State (Option A: Widget layer)
        self.show_camera_labels = True
        self.is_playing = False
        self.loop_enabled = True
        self.speed_multiplier = 1.0

        # Timer for animation
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame)

        # Main 3D view
        self.plotter: QtInteractor = QtInteractor(parent=self)
        self.setCentralWidget(self.plotter)

        # Sync index slider
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(self.view_model.min_index)
        self.slider.setMaximum(self.view_model.max_index)
        self.slider.setValue(self.sync_index)
        self.slider.valueChanged.connect(self._on_sync_index_changed)

        # Playback controls
        self.play_button = QPushButton("▶", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setFixedWidth(30)
        self.play_button.setToolTip("Play/Pause")

        self.loop_button = QPushButton("🔁", self)
        self.loop_button.setCheckable(True)
        self.loop_button.setChecked(True)
        self.loop_button.clicked.connect(self._toggle_loop)
        self.loop_button.setFixedWidth(30)
        self.loop_button.setToolTip("Toggle loop")

        self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.speed_slider.setMinimum(1)  # 0.1x
        self.speed_slider.setMaximum(30)  # 3.0x
        self.speed_slider.setValue(10)  # 1.0x default
        self.speed_slider.setFixedWidth(80)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self.speed_slider.setToolTip("Playback speed")

        self.speed_label = QLabel("1.0x", self)
        self.speed_label.setFixedWidth(35)

        # Camera labels checkbox
        self.labels_checkbox = QCheckBox("Camera Labels", self)
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self._on_labels_toggled)

        # Layout containers for status bar
        playback_widget = QWidget(self)
        playback_layout = QHBoxLayout(playback_widget)
        playback_layout.setContentsMargins(0, 0, 0, 0)
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.loop_button)
        playback_layout.addWidget(self.speed_slider)
        playback_layout.addWidget(self.speed_label)

        right_widget = QWidget(self)
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.labels_checkbox)
        right_layout.addWidget(self.slider)

        # Add to status bar
        self.statusBar().addWidget(playback_widget, stretch=1)
        self.statusBar().addPermanentWidget(right_widget)

        self._initialize_scene()
        self._create_camera_actors()
        self._on_sync_index_changed(self.sync_index)  # Initial render

    def _initialize_scene(self):
        self.plotter.show_axes()
        self.plotter.camera_position = [(4, 4, 4), (0, 0, 0), (0, 0, 1)]
        self.plotter.show_grid()
        self.plotter.enable_terrain_style()
        logger.info("PyVista scene initialized")

    def _create_camera_actors(self):
        """Create static camera mesh actors."""
        camera_geom = self.view_model.get_camera_geometry()

        if camera_geom is None:
            logger.warning("No camera geometry available")
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

        # Store label actor reference for visibility control
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
        logger.info(f"Added {len(camera_geom['labels'])} cameras to scene")

    def _on_sync_index_changed(self, sync_index: int):
        """Update all dynamic geometry when slider moves."""
        self.sync_index = sync_index

        # Update point cloud
        point_geom = self.view_model.get_point_geometry(sync_index)
        if point_geom is None:
            points = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.float32)
        else:
            points, colors = point_geom

        cloud = pv.PolyData(points)
        cloud.point_data["colors"] = (colors * 255).astype(np.uint8)

        self.plotter.add_mesh(
            cloud,
            name="mediapipe_points",
            render_points_as_spheres=True,
            point_size=5,
            scalars="colors",
            rgb=True,
            reset_camera=False,
        )

        # Update wireframe
        wireframe_geom = self.view_model.get_wireframe_geometry(sync_index)
        if wireframe_geom is None:
            dummy_points = np.empty((0, 3), dtype=np.float32)
            dummy_lines = np.empty(0, dtype=np.int32)
            wireframe = pv.PolyData(dummy_points, lines=dummy_lines)
        else:
            points, lines, colors = wireframe_geom
            wireframe = pv.PolyData(points, lines=lines)
            wireframe.cell_data["colors"] = (colors * 255).astype(np.uint8)

        self.plotter.add_mesh(
            wireframe,
            name="wireframe",
            scalars="colors" if wireframe_geom is not None else None,
            rgb=True,
            render_lines_as_tubes=True,
            line_width=0.005,
            reset_camera=False,
        )

        self.plotter.render()
        logger.debug(f"Updated geometry for sync_index {sync_index}")

    def _toggle_playback(self, checked: bool):
        """Start/stop animation timer."""
        self.is_playing = checked
        if checked:
            self.play_button.setText("⏸")
            self._start_playback()
        else:
            self.play_button.setText("▶")
            self.playback_timer.stop()

    def _start_playback(self):
        """Calculate timer interval and start animation."""
        if self.view_model.frame_rate is None or self.view_model.frame_rate <= 0:
            logger.warning("Invalid frame rate, cannot start playback")
            self.play_button.setChecked(False)
            return

        interval_ms = int(1000 / (self.view_model.frame_rate * self.speed_multiplier))
        self.playback_timer.start(interval_ms)

    def _advance_frame(self):
        """Advance to next frame, handling loop/end of range."""
        next_index = self.sync_index + 1

        if next_index > self.view_model.max_index:
            if self.loop_enabled:
                next_index = self.view_model.min_index
            else:
                self.play_button.setChecked(False)  # Stop at end
                return

        self.slider.setValue(next_index)  # This triggers _on_sync_index_changed

    def _toggle_loop(self, checked: bool):
        """Toggle loop behavior."""
        self.loop_enabled = checked
        self.loop_button.setChecked(checked)

    def _on_speed_changed(self, value: int):
        """Update speed multiplier from slider (1-30 maps to 0.1-3.0)."""
        self.speed_multiplier = value / 10.0
        self.speed_label.setText(f"{self.speed_multiplier:.1f}x")

        # Restart timer if playing to apply new speed
        if self.is_playing:
            self._start_playback()

    def _on_labels_toggled(self, state: int):
        """Show/hide camera labels."""
        self.show_camera_labels = state == Qt.CheckState.Checked.value
        if hasattr(self, "_label_actor"):
            self._label_actor.SetVisibility(self.show_camera_labels)
            self.plotter.render()


# --- Demo script ---
if __name__ == "__main__":
    import sys
    import os

    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    import caliscope.persistence as persistence
    from caliscope import __root__
    from caliscope.ui.viz.playback_view_model import PlaybackViewModel
    from caliscope.ui.viz.wireframe_loader import load_wireframe_config

    # Load test data
    session_path = __root__ / "tests" / "sessions" / "4_cam_recording"
    xyz_path = session_path / "recordings" / "recording_1" / "HOLISTIC" / "xyz_HOLISTIC.csv"
    camera_array_path = session_path / "camera_array.toml"

    world_points = persistence.load_world_points_csv(xyz_path)
    camera_array = persistence.load_camera_array(camera_array_path)

    # Load wireframe
    wireframe_path = __root__ / "src" / "caliscope" / "ui" / "viz" / "wireframes" / "holistic_wireframe.toml"
    wireframe_config = load_wireframe_config(wireframe_path)

    # Add frame_rate to ViewModel temporarily for demo
    view_model = PlaybackViewModel(
        world_points=world_points,
        camera_array=camera_array,
        wireframe_segments=wireframe_config.segments,
    )

    widget = PlaybackTriangulationWidgetPyVista(view_model=view_model)
    widget.show()
    widget.resize(900, 650)  # Wider to accommodate controls

    sys.exit(app.exec())
