# Tests for CoverageHeatmapWidget
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestConstruction:
    def test_construction_with_no_data(self, qapp):
        from caliscope.synthetic.explorer.widgets import CoverageHeatmapWidget

        widget = CoverageHeatmapWidget()
        assert widget._coverage is None
        assert widget._killed_linkages == set()


class TestSetData:
    def test_set_data_updates_coverage(self, qapp):
        from caliscope.synthetic.explorer.widgets import CoverageHeatmapWidget

        widget = CoverageHeatmapWidget()
        coverage = np.array([[10, 5], [5, 8]], dtype=np.int64)
        widget.set_data(coverage, set())
        assert widget._coverage is not None
        assert np.array_equal(widget._coverage, coverage)


class TestLinkageState:
    def test_is_linkage_killed_returns_true_for_killed(self, qapp):
        from caliscope.synthetic.explorer.widgets import CoverageHeatmapWidget

        widget = CoverageHeatmapWidget()
        coverage = np.array([[10, 5], [5, 8]], dtype=np.int64)
        widget.set_data(coverage, {(0, 1)})
        assert widget._is_linkage_killed(0, 1)
        assert widget._is_linkage_killed(1, 0)


class TestColorCalculation:
    def test_diagonal_cell_is_blue(self, qapp):
        from caliscope.synthetic.explorer.widgets import CoverageHeatmapWidget

        widget = CoverageHeatmapWidget()
        color = widget._cell_color(0, 0, 10, False, 5)
        assert color.blue() > color.red()
