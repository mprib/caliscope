from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


def setup_spinbox_sizing(
    spin_box: QSpinBox | QDoubleSpinBox,
    min_value: int,
    max_value: int,
    padding: int | None = None,
) -> None:
    """Center the text, set the range, and make the box wide enough for its longest value.

    padding covers the spin arrows and frame; it defaults to the font height plus 20px.
    """
    spin_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    spin_box.setMinimum(min_value)
    spin_box.setMaximum(max_value)

    if isinstance(spin_box, QDoubleSpinBox):
        decimals = spin_box.decimals()
        extremes = (f"{spin_box.minimum():.{decimals}f}", f"{spin_box.maximum():.{decimals}f}")
    else:
        extremes = (str(spin_box.minimum()), str(spin_box.maximum()))

    fm = spin_box.fontMetrics()
    if padding is None:
        padding = fm.height() + 20
    spin_box.setMinimumWidth(max(fm.horizontalAdvance(text) for text in extremes) + padding)
