"""CharucoConfigPanel round-trip: thickness has no widget and must ride the
params cache — otherwise any GUI edit would silently zero a TOML-set value."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from caliscope.core.charuco import Charuco
from caliscope.gui.widgets.charuco_config_panel import CharucoConfigPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_thickness_survives_gui_edit(qapp):
    thick = Charuco.from_squares(columns=4, rows=5, square_size_cm=5.0, thickness_cm=0.6)
    panel = CharucoConfigPanel(initial_charuco=thick)

    panel._row_spin.setValue(panel._row_spin.value() + 1)

    result = panel.get_charuco()
    assert result.thickness_cm == 0.6
    assert result.rows == thick.rows + 1


def test_set_values_carries_thickness(qapp):
    """The same-as-intrinsic sync path: set_values refreshes the cache."""
    thin = Charuco.from_squares(columns=4, rows=5, square_size_cm=5.0)
    panel = CharucoConfigPanel(initial_charuco=thin)

    thick = Charuco.from_squares(columns=4, rows=5, square_size_cm=5.0, thickness_cm=0.6)
    panel.set_values(thick)

    assert panel.get_charuco().thickness_cm == 0.6
