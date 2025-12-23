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
        self._create_point_actor()

    def _initialize_scene(self):
        self.plotter.show_axes()
        self.plotter.camera_position = [(4, 4, 4), (0, 0, 0), (0, 0, 1)]
        self.plotter.show_grid()
        self.plotter.enable_terrain_style()
        logger.info("PyVista scene initialized")

    def _create_point_actor(self):
        """Create empty point actor for updates."""
        # Start with single point at origin
        dummy_points = np.array([[0, 0, 0]], dtype=np.float32)
        self.point_cloud = pv.PolyData(dummy_points)
        self.point_cloud.point_data["colors"] = np.array([[0.9, 0.9, 0.9]], dtype=np.float32)

        self.point_actor = self.plotter.add_mesh(
            self.point_cloud,
            render_points_as_spheres=True,
            point_size=5,
            scalars="colors",
            rgb=True,
        )

    def _on_sync_index_changed(self, sync_index: int):
        """Update point geometry when slider moves."""
        self.sync_index = sync_index

        # Get geometry from ViewModel
        point_geom = self.view_model.get_point_geometry(sync_index)

        if point_geom is None:
            # No data for this frame - hide points
            self.point_cloud.points = np.empty((0, 3))
            self.point_cloud.point_data["colors"] = np.empty((0, 3))
        else:
            positions, colors = point_geom
            self.point_cloud.points = positions
            self.point_cloud.point_data["colors"] = colors

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

    session_path = __root__ / "tests" / "sessions" / "4_cam_recording"
    xyz_path = session_path / "recordings" / "recording_1" / "HOLISTIC" / "xyz_HOLISTIC.csv"
    camera_array_path = session_path / "camera_array.toml"

    world_points = persistence.load_world_points_csv(xyz_path)
    camera_array = persistence.load_camera_array(camera_array_path)

    view_model = PlaybackViewModel(
        world_points=world_points,
        camera_array=camera_array,
        wireframe_segments=None,
    )

    # Set slider range based on actual data
    min_idx = world_points.df["sync_index"].min()
    max_idx = world_points.df["sync_index"].max()

    widget = PlaybackTriangulationWidgetPyVista(view_model=view_model)
    widget.set_sync_index_range(min_idx, max_idx)
    widget.show()
    widget.resize(800, 600)

    sys.exit(app.exec())
