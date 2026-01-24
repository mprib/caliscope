"""Four-panel synchronized 3D visualization of calibration pipeline stages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from caliscope.ui.viz.playback_triangulation_widget_pyvista import (
    PlaybackTriangulationWidgetPyVista,
)
from caliscope.ui.viz.playback_view_model import PlaybackViewModel

if TYPE_CHECKING:
    from caliscope.synthetic.explorer.presenter import PipelineResult
    from caliscope.synthetic.scene import SyntheticScene

logger = logging.getLogger(__name__)


# Camera scale for synthetic scenes (mm units, ~4000mm extent)
SYNTHETIC_CAMERA_SCALE = 0.25


class StoryboardView(QWidget):
    """Four-panel synchronized visualization of calibration pipeline stages.

    Layout:
    +------------------+------------------+
    |  GROUND TRUTH    |   BOOTSTRAPPED   |
    +------------------+------------------+
    |    OPTIMIZED     |     ALIGNED      |
    +------------------+------------------+

    All panels have synchronized camera rotation/pan/zoom.
    Frame navigation is controlled externally via set_frame().
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._panels: dict[str, PlaybackTriangulationWidgetPyVista] = {}
        self._view_models: dict[str, PlaybackViewModel | None] = {
            "ground_truth": None,
            "bootstrapped": None,
            "optimized": None,
            "aligned": None,
        }
        self._sync_in_progress = False
        self._current_frame = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        panel_configs = [
            ("ground_truth", "GROUND TRUTH", 0, 0),
            ("bootstrapped", "BOOTSTRAPPED", 0, 1),
            ("optimized", "OPTIMIZED", 1, 0),
            ("aligned", "ALIGNED", 1, 1),
        ]

        for key, title, row, col in panel_configs:
            panel = self._create_panel(key, title)
            layout.addWidget(panel, row, col)

        # Equal stretch for all cells
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

    def _create_panel(self, key: str, title: str) -> QWidget:
        """Create a labeled panel placeholder."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold; font-size: 12px; color: white;")
        layout.addWidget(label)

        # Placeholder - will be replaced when view model is set
        placeholder = QLabel("No data")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("background-color: #333; color: #666;")
        placeholder.setMinimumSize(200, 150)
        layout.addWidget(placeholder, stretch=1)

        # Store reference to placeholder for later replacement
        container.setProperty("placeholder", placeholder)
        container.setProperty("layout_ref", layout)
        container.setProperty("key", key)

        return container

    def set_scene(self, scene: SyntheticScene) -> None:
        """Set ground truth from synthetic scene."""
        vm = PlaybackViewModel(
            camera_array=scene.camera_array,
            world_points=scene.world_points,
        )
        self._set_panel_view_model("ground_truth", vm)

        # Clear other panels when scene changes
        for key in ["bootstrapped", "optimized", "aligned"]:
            self._clear_panel(key)

    def set_result(self, result: PipelineResult) -> None:
        """Update panels with pipeline result."""
        # Bootstrapped
        if result.bootstrapped_cameras and result.bootstrapped_world_points:
            vm = PlaybackViewModel(
                camera_array=result.bootstrapped_cameras,
                world_points=result.bootstrapped_world_points,
            )
            self._set_panel_view_model("bootstrapped", vm)

        # Optimized
        if result.optimized_cameras and result.optimized_world_points:
            vm = PlaybackViewModel(
                camera_array=result.optimized_cameras,
                world_points=result.optimized_world_points,
            )
            self._set_panel_view_model("optimized", vm)

        # Aligned
        if result.aligned_cameras and result.aligned_world_points:
            vm = PlaybackViewModel(
                camera_array=result.aligned_cameras,
                world_points=result.aligned_world_points,
            )
            self._set_panel_view_model("aligned", vm)

    def set_frame(self, frame: int) -> None:
        """Set current frame on all panels."""
        self._current_frame = frame
        for panel in self._panels.values():
            panel.set_sync_index(frame)

    def _set_panel_view_model(self, key: str, view_model: PlaybackViewModel) -> None:
        """Replace panel content with a PlaybackTriangulationWidget."""
        # Find the container widget
        container = self._find_container(key)
        if container is None:
            logger.warning(f"Could not find container for key: {key}")
            return

        # Use layout_ref to avoid shadowing built-in 'layout'
        panel_layout: QVBoxLayout | None = container.property("layout_ref")
        placeholder: QLabel | None = container.property("placeholder")

        if panel_layout is None:
            logger.warning(f"No layout_ref property found for container: {key}")
            return

        # Remove existing panel if present
        if key in self._panels:
            old_panel = self._panels[key]
            panel_layout.removeWidget(old_panel)
            old_panel.deleteLater()
            del self._panels[key]

        # Remove placeholder if present
        if placeholder is not None and placeholder.parent() is not None:
            panel_layout.removeWidget(placeholder)
            placeholder.hide()

        # Create new panel
        panel = PlaybackTriangulationWidgetPyVista(
            view_model,
            camera_scale=SYNTHETIC_CAMERA_SCALE,
        )
        panel.show_playback_controls(False)
        panel_layout.addWidget(panel, stretch=1)

        self._panels[key] = panel
        self._view_models[key] = view_model

        # Connect camera sync
        self._connect_camera_sync(panel)

        # Set current frame
        panel.set_sync_index(self._current_frame)

    def _clear_panel(self, key: str) -> None:
        """Clear a panel back to placeholder state."""
        container = self._find_container(key)
        if container is None:
            return

        panel_layout: QVBoxLayout | None = container.property("layout_ref")
        placeholder: QLabel | None = container.property("placeholder")

        if panel_layout is None:
            return

        # Remove existing panel
        if key in self._panels:
            old_panel = self._panels[key]
            panel_layout.removeWidget(old_panel)
            old_panel.deleteLater()
            del self._panels[key]

        # Show placeholder
        if placeholder is not None:
            panel_layout.addWidget(placeholder, stretch=1)
            placeholder.show()

        self._view_models[key] = None

    def _find_container(self, key: str) -> QWidget | None:
        """Find container widget by key."""
        grid_layout = self.layout()
        if grid_layout is None:
            return None

        for i in range(grid_layout.count()):
            item = grid_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if widget.property("key") == key:
                    return widget
        return None

    def _connect_camera_sync(self, panel: PlaybackTriangulationWidgetPyVista) -> None:
        """Connect VTK observer for camera synchronization."""
        if hasattr(panel, "plotter") and panel.plotter is not None:
            if hasattr(panel.plotter, "iren") and panel.plotter.iren is not None:
                vtk_iren = panel.plotter.iren.interactor
                if vtk_iren is not None:
                    vtk_iren.AddObserver(
                        "EndInteractionEvent",
                        lambda obj, event, p=panel: self._sync_camera_from(p),
                    )

    def _sync_camera_from(self, source: PlaybackTriangulationWidgetPyVista) -> None:
        """Copy camera state from source to all other panels."""
        if self._sync_in_progress:
            return

        self._sync_in_progress = True
        try:
            cam = source.plotter.camera
            for panel in self._panels.values():
                if panel is not source:
                    panel.plotter.camera.position = cam.position
                    panel.plotter.camera.focal_point = cam.focal_point
                    panel.plotter.camera.up = cam.up
                    panel.plotter.render()
        finally:
            self._sync_in_progress = False

    def suspend_vtk(self) -> None:
        """Pause VTK interactors to reduce CPU when widget is not visible."""
        for panel in self._panels.values():
            panel.suspend_vtk()

    def resume_vtk(self) -> None:
        """Resume VTK interactors when widget becomes visible."""
        for panel in self._panels.values():
            panel.resume_vtk()

    def cleanup(self) -> None:
        """Explicit cleanup - MUST be called before destruction."""
        self.suspend_vtk()

        for key in list(self._panels.keys()):
            panel = self._panels.pop(key)
            panel.deleteLater()

        self._view_models = {k: None for k in self._view_models}

    def closeEvent(self, event) -> None:
        """Handle close event - defensive cleanup."""
        self.cleanup()
        super().closeEvent(event)
