"""
PyVista-based 3D visualization widget for extrinsic calibration results.

Displays camera frustums and triangulated charuco points, with controls for:
- Navigating through sync indices (slider)
- Rotating the coordinate system (6 rotation buttons)
- Setting the world origin to a charuco board position
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyvista as pv
from PySide6.QtCore import Qt, Signal
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

from caliscope.ui.viz.playback_view_model import PlaybackViewModel

logger = logging.getLogger(__name__)


class ExtrinsicCalibrationWidget(QWidget):
    """
    PyVista-based widget for visualizing and adjusting extrinsic calibration.

    Emits signals for user actions - does NOT call coordinator directly.
    This follows MVP pattern where the container wires signals to coordinator methods.

    Note: Inherits from QWidget (not QMainWindow) so it can be embedded in layouts.
    """

    # Signals for user actions (container wires these to coordinator)
    rotation_requested = Signal(str, float)  # (axis, angle_degrees)
    set_origin_requested = Signal(int)  # sync_index

    def __init__(self, view_model: PlaybackViewModel, parent: QWidget | None = None):
        super().__init__(parent)

        # Lazy import to ensure QApplication exists before pyvistaqt init
        from pyvistaqt import QtInteractor

        self.view_model = view_model
        self.sync_index: int = self.view_model.min_index

        # UI state
        self.show_camera_labels = True

        # Persistent mesh references
        self._point_cloud_mesh: pv.PolyData | None = None
        self._label_actor = None

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

        # Initialize scene
        self._initialize_scene()
        self._create_static_actors()
        self._create_dynamic_actors()
        self._on_sync_index_changed(self.sync_index)

    def _create_controls(self) -> QWidget:
        """Create control bar with rotation buttons, slider, and set origin button."""
        controls = QWidget(self)
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(5, 5, 5, 5)

        # Rotation buttons - organized by axis
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

            layout.addWidget(btn_pos)
            layout.addWidget(btn_neg)

        # Set Origin button
        self.set_origin_btn = QPushButton("Set Origin", self)
        self.set_origin_btn.setToolTip("Set world origin to board position at current frame")
        self.set_origin_btn.clicked.connect(self._on_set_origin_clicked)
        layout.addWidget(self.set_origin_btn)

        # Camera labels toggle
        self.labels_checkbox = QCheckBox("Labels", self)
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self._on_labels_toggled)
        layout.addWidget(self.labels_checkbox)

        layout.addStretch()

        # Frame label and slider on right side
        self.index_label = QLabel(f"Frame: {self.sync_index}", self)
        self.index_label.setFixedWidth(80)
        layout.addWidget(self.index_label)

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setMinimum(self.view_model.min_index)
        self._slider.setMaximum(self.view_model.max_index)
        self._slider.setValue(self.sync_index)
        self._slider.valueChanged.connect(self._on_sync_index_changed)
        layout.addWidget(self._slider, stretch=1)

        return controls

    def _initialize_scene(self):
        """Set up PyVista scene defaults."""
        self.plotter.show_axes()
        self.plotter.camera_position = [(4, 4, 4), (0, 0, 0), (0, 0, 1)]
        self.plotter.show_grid()
        self.plotter.enable_trackball_style()
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

    def showEvent(self, event) -> None:
        """Resume interactor and render when widget becomes visible.

        Pair with hideEvent to reduce idle CPU when widget is hidden.
        VTK's interactor runs a timer that polls for events continuously;
        we disable it when hidden to avoid wasted CPU cycles.
        """
        super().showEvent(event)
        # Re-enable interactor event processing
        if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
            self.plotter.iren.EnableRenderOn()
        self.plotter.render()

    def hideEvent(self, event) -> None:
        """Pause interactor when widget is hidden to reduce idle CPU.

        VTK's RenderWindowInteractor runs a repeating timer (~10ms) that
        continuously polls for events. When the tab is switched away,
        this timer keeps running, wasting CPU. Disabling render while
        hidden eliminates this overhead.
        """
        super().hideEvent(event)
        # Disable render requests while hidden
        if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
            self.plotter.iren.EnableRenderOff()

    def set_view_model(self, view_model: PlaybackViewModel) -> None:
        """
        Replace the ViewModel and rebuild the scene.

        Called by container when the underlying bundle changes (after rotation/set origin).
        """
        self.view_model = view_model

        # Update slider range
        self._slider.setMinimum(view_model.min_index)
        self._slider.setMaximum(view_model.max_index)

        # Clamp current index to valid range
        if self.sync_index < view_model.min_index:
            self.sync_index = view_model.min_index
        elif self.sync_index > view_model.max_index:
            self.sync_index = view_model.max_index

        self._slider.setValue(self.sync_index)

        # Rebuild scene with new geometry
        self.plotter.clear()
        self._initialize_scene()
        self._create_static_actors()
        self._create_dynamic_actors()
        self._on_sync_index_changed(self.sync_index)
