class CalibrationError(Exception):
    """Raised when a calibration operation fails.

    Messages are actionable -- they explain what went wrong and what to do.
    """

    pass


class CalibrationWarning(UserWarning):
    """Warning for non-fatal calibration issues."""

    pass
