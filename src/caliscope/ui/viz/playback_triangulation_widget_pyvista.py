# --- File: src/caliscope/ui/viz/playback_triangulation_widget_pyvista.py ---

import logging
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider, QMainWindow
from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np

from caliscope.ui.viz.playback_view_model import PlaybackViewModel

logger = logging.getLogger(__name__)


class PlaybackTriangulationWidgetPyVista(QMainWindow):
    def __init__(self, view_model: PlaybackViewModel, parent=None):
        super().__init__(parent)
        self.view_model = view_model
        self.sync_index = 0

        self.plotter = QtInteractor(parent=self)
        self.setCentralWidget(self.plotter)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.statusBar().addWidget(self.slider, stretch=1)

        self.slider.valueChanged.connect(self._on_sync_index_changed)

        self._initialize_scene()

    def _initialize_scene(self):
        self.plotter.show_axes()
        self.plotter.camera_position = [(4, 4, 4), (0, 0, 0), (0, 0, 1)]
        self.plotter.show_grid()
        self.plotter.enable_terrain_style()
        logger.info("PyVista scene initialized")

    def _on_sync_index_changed(self, sync_index: int):
        """Update point geometry when slider moves."""
        self.sync_index = sync_index

        point_geom = self.view_model.get_point_geometry(sync_index)

        if point_geom is None:
            # Clear points by setting empty array
            points = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.float32)
        else:
            points, colors = point_geom

        # Create PolyData
        cloud = pv.PolyData(points)
        cloud.point_data["colors"] = (colors * 255).astype(np.uint8)

        # Update or add mesh with name (no flicker)
        self.plotter.add_mesh(
            cloud,
            name="mediapipe_points",  # Key: updates existing actor
            render_points_as_spheres=True,
            point_size=5,
            scalars="colors",
            rgb=True,
            reset_camera=False,  # Don't jump camera
        )

        self.plotter.render()
        logger.debug(f"Updated points for sync_index {sync_index}")

    def set_sync_index_range(self, min_index: int, max_index: int):
        self.slider.setMinimum(min_index)
        self.slider.setMaximum(max_index)
        logger.info(f"Sync index range set to [{min_index}, {max_index}]")


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

    view_model = PlaybackViewModel(
        world_points=world_points,
        camera_array=camera_array,
        wireframe_segments=wireframe_config.segments,
    )

    # Create widget and set range
    widget = PlaybackTriangulationWidgetPyVista(view_model=view_model)
    widget.set_sync_index_range(world_points.df["sync_index"].min(), world_points.df["sync_index"].max())

    widget.show()
    widget.resize(800, 600)

    sys.exit(app.exec())
