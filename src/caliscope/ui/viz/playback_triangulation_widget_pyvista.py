import logging

import pyvista as pv
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QWidget,
)
from pyvistaqt import QtInteractor

from caliscope.ui.viz.playback_view_model import PlaybackViewModel

logger = logging.getLogger(__name__)


class PlaybackTriangulationWidgetPyVista(QMainWindow):
    def __init__(self, view_model: PlaybackViewModel, parent: QWidget | None = None):
        super().__init__(parent)
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

        # Timer for animation
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_frame)

        # Main 3D view
        self.plotter: QtInteractor = QtInteractor(parent=self)
        self.setCentralWidget(self.plotter)

        # --- Standard UI Setup ---
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(self.view_model.min_index)
        self.slider.setMaximum(self.view_model.max_index)
        self.slider.setValue(self.sync_index)
        self.slider.valueChanged.connect(self._on_sync_index_changed)

        self.play_button = QPushButton("▶", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setFixedWidth(30)

        self.loop_button = QPushButton("🔁", self)
        self.loop_button.setCheckable(True)
        self.loop_button.setChecked(True)
        self.loop_button.clicked.connect(self._toggle_loop)
        self.loop_button.setFixedWidth(30)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(30)
        self.speed_slider.setValue(10)
        self.speed_slider.setFixedWidth(80)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        self.speed_label = QLabel("1.0x", self)
        self.speed_label.setFixedWidth(35)

        self.labels_checkbox = QCheckBox("Camera Labels", self)
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self._on_labels_toggled)

        # Layouts
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

        self.statusBar().addWidget(playback_widget, stretch=1)
        self.statusBar().addPermanentWidget(right_widget)

        # --- Initialization ---
        self._initialize_scene()
        self._create_static_actors()
        self._create_dynamic_actors()  # Create the persistent meshes once
        self._on_sync_index_changed(self.sync_index)  # Render first frame

    def _initialize_scene(self):
        self.plotter.show_axes()
        self.plotter.camera_position = [(4, 4, 4), (0, 0, 0), (0, 0, 1)]
        self.plotter.show_grid()
        self.plotter.enable_terrain_style()
        logger.info("PyVista scene initialized")

    def _create_static_actors(self):
        """Create static camera geometry."""
        camera_geom = self.view_model.get_camera_geometry()
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

        self.plotter.add_mesh(
            self._wireframe_mesh,
            name="wireframe",
            scalars="colors" if len(line_colors) > 0 else None,
            rgb=True,
            render_lines_as_tubes=True,
            line_width=5,
            reset_camera=False,
        )

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
            self.play_button.setText("⏸")
            self._start_playback()
        else:
            self.play_button.setText("▶")
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
