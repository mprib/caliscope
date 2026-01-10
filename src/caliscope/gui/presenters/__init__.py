"""Presenters for MVP architecture.

Presenters coordinate workflow state and adapt domain logic for Views.
They hold transient "scratchpad" state; results aren't persisted until
emitted to the Controller.
"""

from caliscope.gui.presenters.intrinsic_calibration_presenter import (
    IntrinsicCalibrationPresenter,
    PresenterState,
)

__all__ = ["IntrinsicCalibrationPresenter", "PresenterState"]
