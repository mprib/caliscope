"""
Container that wires ExtrinsicCalibrationWidget signals to WorkspaceCoordinator.

This is the thin glue layer that connects the pure UI widget (which only emits signals)
to the application coordinator (which manages domain state and persistence).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QVBoxLayout, QWidget

from caliscope.ui.viz.extrinsic_calibration_widget import ExtrinsicCalibrationWidget
from caliscope.ui.viz.playback_view_model import PlaybackViewModel

if TYPE_CHECKING:
    from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)


class ExtrinsicCalibrationTab(QWidget):
    """
    Container that wires ExtrinsicCalibrationWidget to WorkspaceCoordinator.

    Responsibilities:
    - Create widget with ViewModel built from coordinator's bundle
    - Wire widget signals to coordinator methods
    - Listen for bundle changes and refresh the widget's ViewModel
    """

    def __init__(self, coordinator: WorkspaceCoordinator, parent: QWidget | None = None):
        super().__init__(parent)
        self._coordinator = coordinator
        self._widget: ExtrinsicCalibrationWidget | None = None

        # Layout to hold the widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Initial setup if bundle exists
        self._refresh_widget()

        # Listen for bundle changes from coordinator
        self._coordinator.bundle_changed.connect(self._on_bundle_changed)

    def _refresh_widget(self) -> None:
        """Create or update widget with current bundle data."""
        bundle = self._coordinator.point_data_bundle

        if bundle is None:
            logger.warning("No point data bundle available for extrinsic calibration tab")
            return

        # Create ViewModel from bundle
        view_model = PlaybackViewModel(
            world_points=bundle.world_points,
            camera_array=bundle.camera_array,
            wireframe_segments=None,  # No skeleton for charuco points
        )

        if self._widget is None:
            # First time - create widget
            self._widget = ExtrinsicCalibrationWidget(view_model, parent=self)
            self.layout().addWidget(self._widget)

            # Wire signals to coordinator
            self._widget.rotation_requested.connect(self._on_rotation_requested)
            self._widget.set_origin_requested.connect(self._on_set_origin_requested)

            logger.info("ExtrinsicCalibrationWidget created and wired")
        else:
            # Update existing widget with new ViewModel
            self._widget.set_view_model(view_model)
            logger.info("ExtrinsicCalibrationWidget ViewModel refreshed")

    def _on_rotation_requested(self, axis: str, angle_degrees: float) -> None:
        """Handle rotation request from widget."""
        logger.info(f"Rotation requested: axis={axis}, angle={angle_degrees}")
        self._coordinator.rotate_bundle(axis, angle_degrees)

    def _on_set_origin_requested(self, sync_index: int) -> None:
        """Handle set origin request from widget."""
        logger.info(f"Set origin requested: sync_index={sync_index}")
        self._coordinator.set_bundle_origin(sync_index)

    def _on_bundle_changed(self) -> None:
        """Handle bundle change signal from coordinator."""
        logger.info("Bundle changed, refreshing widget")
        self._refresh_widget()
