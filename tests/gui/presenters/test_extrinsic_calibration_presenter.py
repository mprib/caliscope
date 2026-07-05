"""Tests for ExtrinsicCalibrationPresenter's origin-option labeling.

The "board" label used to fall out of `not constraints` being true only for
charuco (which passed constraints=None). Now that charuco calibration also
produces constraints (ConstraintSet.from_charuco), that condition is dead;
labeling is driven by the target type the coordinator hands the presenter.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QCoreApplication

from caliscope.cameras.camera_array import CameraArray
from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    ExtrinsicCalibrationPresenter,
)


@pytest.fixture
def qapp():
    """Ensure QCoreApplication exists for Qt signal tests."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


def _make_presenter(extrinsic_target_type=None) -> ExtrinsicCalibrationPresenter:
    return ExtrinsicCalibrationPresenter(
        task_manager=MagicMock(),
        camera_array=CameraArray({}),
        image_points_path=Path("does_not_exist.csv"),
        extrinsic_target_type=extrinsic_target_type,
    )


def test_is_board_origin_true_for_charuco_target(qapp):
    """Charuco always labels its single non-static object 'board', regardless
    of object/static counts, once the target type is known."""
    presenter = _make_presenter(extrinsic_target_type="charuco")
    assert presenter._is_board_origin(object_ids=[0], static_ids=frozenset()) is True
    # Still "board" even if a static id were somehow present (charuco never
    # produces one, but the label should track the known target type).
    assert presenter._is_board_origin(object_ids=[0], static_ids=frozenset({0})) is True


def test_is_board_origin_false_for_aruco_target(qapp):
    """A single-marker ArUco target is known precisely and labeled 'marker N',
    not 'board', now that the target type is available."""
    presenter = _make_presenter(extrinsic_target_type="aruco")
    assert presenter._is_board_origin(object_ids=[0], static_ids=frozenset()) is False


def test_is_board_origin_falls_back_to_heuristic_when_target_type_unknown(qapp):
    """Without a known target type (e.g. a presenter built directly in a
    test), fall back to the old heuristic: exactly one object, no statics."""
    presenter = _make_presenter(extrinsic_target_type=None)
    assert presenter._is_board_origin(object_ids=[0], static_ids=frozenset()) is True
    assert presenter._is_board_origin(object_ids=[0, 1], static_ids=frozenset()) is False
    assert presenter._is_board_origin(object_ids=[0], static_ids=frozenset({0})) is False
