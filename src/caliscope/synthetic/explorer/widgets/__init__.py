"""Widget components for the Synthetic Calibration Explorer."""

from caliscope.gui.widgets import CoverageHeatmapWidget  # Import from new location
from caliscope.synthetic.explorer.widgets.per_camera_view import PerCameraObservationsView
from caliscope.synthetic.explorer.widgets.storyboard_view import StoryboardView

__all__ = [
    "CoverageHeatmapWidget",
    "PerCameraObservationsView",
    "StoryboardView",
]
