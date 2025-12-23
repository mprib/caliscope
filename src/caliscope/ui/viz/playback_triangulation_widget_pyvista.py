# --- File: src/caliscope/ui/viz/playback_triangulation_widget_pyvista.py ---

import logging
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider, QMainWindow
from pyvistaqt import QtInteractor

from caliscope.ui.viz.playback_view_model import PlaybackViewModel

logger = logging.getLogger(__name__)


class PlaybackTriangulationWidgetPyVista(QMainWindow):
    def __init__(self, view_model: PlaybackViewModel, parent=None):
        super().__init__(parent)
        self.view_model = view_model
        self.sync_index = 0

        # Create plotter with self as parent
        self.plotter = QtInteractor(parent=self)
        self.setCentralWidget(self.plotter)

        # Add slider to status bar
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

        # Fix camera controls: turntable style keeps horizon level
        self.plotter.enable_terrain_style()

        logger.info("PyVista scene initialized")

    def _on_sync_index_changed(self, sync_index: int):
        self.sync_index = sync_index
        logger.debug(f"Sync index changed to {sync_index}")


# --- Demo script ---
if __name__ == "__main__":
    import sys
    import os

    # Force X11 on Wayland
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

    widget = PlaybackTriangulationWidgetPyVista(view_model=view_model)
    widget.show()
    widget.resize(800, 600)

    sys.exit(app.exec())
