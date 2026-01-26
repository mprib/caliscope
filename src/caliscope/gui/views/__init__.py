"""Views for MVP architecture.

Views handle rendering and user input. They receive display-ready data
from Presenters and delegate actions back via method calls.
"""

from caliscope.gui.views.extrinsic_calibration_view import ExtrinsicCalibrationView

__all__ = ["ExtrinsicCalibrationView"]
