"""
PyVista-based 3D visualization widget for extrinsic calibration results.

Displays camera frustums and triangulated charuco points, with controls for:
- Navigating through sync indices (slider)
- Rotating the coordinate system (6 rotation buttons)
- Setting the world origin to a charuco board position
"""

import logging

import pyvista as pv
from PySide6.QtCore import Qt, Signal
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


class ExtrinsicCalibrationWidget(QMainWindow):
    """
    PyVista-based widget for visualizing and adjusting extrinsic calibration.

    Emits signals for user actions - does NOT call coordinator directly.
    This follows MVP pattern where the container wires signals to coordinator methods.
    """

    # Signals for user actions (container wires these to coordinator)
    rotation_requested = Signal(str, float)  # (axis, angle_degrees)
    set_origin_requested = Signal(int)  # sync_index

    def __init__(self, view_model: PlaybackViewModel, parent: QWidget | None = None):
        super().__init__(parent)
        self.view_model = view_model
        self.sync_index: int = self.view_model.min_index

        # UI state
        self.show_camera_labels = True

        # Persistent mesh references
        self._point_cloud_mesh: pv.PolyData | None = None
        self._label_actor = None

        # Main 3D view
        self.plotter: QtInteractor = QtInteractor(parent=self)
        self.setCentralWidget(self.plotter)

        # Build UI
        self._create_controls()
        self._initialize_scene()
        self._create_static_actors()
        self._create_dynamic_actors()
        self._on_sync_index_changed(self.sync_index)

    def _create_controls(self):
        """Create rotation buttons, slider, and set origin button."""
        # Sync index slider
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(self.view_model.min_index)
        self.slider.setMaximum(self.view_model.max_index)
        self.slider.setValue(self.sync_index)
        self.slider.valueChanged.connect(self._on_sync_index_changed)

        self.index_label = QLabel(f"Frame: {self.sync_index}", self)
        self.index_label.setFixedWidth(80)

        # Rotation buttons - organized by axis
        rotation_widget = QWidget(self)
        rotation_layout = QHBoxLayout(rotation_widget)
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        rotation_layout.setSpacing(2)

        for axis in ["X", "Y", "Z"]:
            axis_lower = axis.lower()
            # Positive rotation button
            btn_pos = QPushButton(f"{axis}+", self)
            btn_pos.setFixedWidth(32)
            btn_pos.setToolTip(f"Rotate +90° around {axis} axis")
            btn_pos.clicked.connect(lambda _, a=axis_lower: self._on_rotate_clicked(a, 90))

            # Negative rotation button
            btn_neg = QPushButton(f"{axis}-", self)
            btn_neg.setFixedWidth(32)
            btn_neg.setToolTip(f"Rotate -90° around {axis} axis")
            btn_neg.clicked.connect(lambda _, a=axis_lower: self._on_rotate_clicked(a, -90))

            rotation_layout.addWidget(btn_pos)
            rotation_layout.addWidget(btn_neg)

        # Set Origin button
        self.set_origin_btn = QPushButton("Set Origin", self)
        self.set_origin_btn.setToolTip("Set world origin to board position at current frame")
        self.set_origin_btn.clicked.connect(self._on_set_origin_clicked)

        # Camera labels toggle
        self.labels_checkbox = QCheckBox("Labels", self)
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self._on_labels_toggled)

        # Layout: left side has rotations + set origin, right side has slider
        left_widget = QWidget(self)
        left_layout = QHBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(rotation_widget)
        left_layout.addWidget(self.set_origin_btn)
        left_layout.addWidget(self.labels_checkbox)

        right_widget = QWidget(self)
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.index_label)
        right_layout.addWidget(self.slider)

        self.statusBar().addWidget(left_widget)
        self.statusBar().addPermanentWidget(right_widget)

    def _initialize_scene(self):
        """Set up PyVista scene defaults."""
        self.plotter.show_axes()
        self.plotter.camera_position = [(4, 4, 4), (0, 0, 0), (0, 0, 1)]
        self.plotter.show_grid()
        self.plotter.enable_terrain_style()
        logger.info("PyVista scene initialized")

    def _create_static_actors(self):
        """Create static camera frustum geometry."""
        camera_geom = self.view_model.get_camera_geometry()
        if camera_geom is None:
            logger.warning("No camera geometry available (extrinsics not calibrated?)")
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

        # Camera labels
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
        """Initialize persistent point cloud mesh."""
        frame_geom = self.view_model.get_frame_geometry(self.sync_index)

        self._point_cloud_mesh = pv.PolyData(frame_geom.points)
        self._point_cloud_mesh.point_data["colors"] = frame_geom.colors

        self.plotter.add_mesh(
            self._point_cloud_mesh,
            name="charuco_points",
            render_points_as_spheres=True,
            point_size=12,
            scalars="colors",
            rgb=True,
            reset_camera=False,
        )

        logger.info("Dynamic point cloud actor initialized")

    def _on_sync_index_changed(self, sync_index: int):
        """Update point cloud for new sync index."""
        self.sync_index = sync_index
        self.index_label.setText(f"Frame: {sync_index}")

        if self._point_cloud_mesh is None:
            return

        frame_geom = self.view_model.get_frame_geometry(sync_index)
        self._point_cloud_mesh.points = frame_geom.points
        self._point_cloud_mesh.point_data["colors"] = frame_geom.colors

        self.plotter.render()

    def _on_rotate_clicked(self, axis: str, angle: float):
        """Emit rotation request signal."""
        self.rotation_requested.emit(axis, angle)

    def _on_set_origin_clicked(self):
        """Emit set origin request signal."""
        self.set_origin_requested.emit(self.sync_index)

    def _on_labels_toggled(self, state: int):
        """Toggle camera label visibility."""
        self.show_camera_labels = state == Qt.CheckState.Checked.value
        if self._label_actor:
            self._label_actor.SetVisibility(self.show_camera_labels)
            self.plotter.render()

    def set_view_model(self, view_model: PlaybackViewModel) -> None:
        """
        Replace the ViewModel and rebuild the scene.

        Called by container when the underlying bundle changes (after rotation/set origin).
        """
        self.view_model = view_model

        # Update slider range
        self.slider.setMinimum(view_model.min_index)
        self.slider.setMaximum(view_model.max_index)

        # Clamp current index to valid range
        if self.sync_index < view_model.min_index:
            self.sync_index = view_model.min_index
        elif self.sync_index > view_model.max_index:
            self.sync_index = view_model.max_index

        self.slider.setValue(self.sync_index)

        # Rebuild scene with new geometry
        self.plotter.clear()
        self._initialize_scene()
        self._create_static_actors()
        self._create_dynamic_actors()
        self._on_sync_index_changed(self.sync_index)


if __name__ == "__main__":
    # For standalone testing, use: scripts/widget_visualization/test_extrinsic_widget.py
    # Direct execution doesn't work because pyvistaqt is imported at module level,
    # before we can apply the pyside6-essentials compatibility patch.
    print("Run: uv run python scripts/widget_visualization/test_extrinsic_widget.py")
    raise SystemExit(1)
