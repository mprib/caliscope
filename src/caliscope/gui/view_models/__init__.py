"""ViewModels for GUI components.

ViewModels are pure data containers that transform domain data into
display-ready formats. Unlike Presenters, they have no Qt signals
and don't coordinate workflows.
"""

from caliscope.gui.view_models.playback_view_model import (
    FrameGeometry,
    PlaybackViewModel,
)

__all__ = [
    "FrameGeometry",
    "PlaybackViewModel",
]
