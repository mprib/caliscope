"""Tests for ExtrinsicCalibrationPresenter's origin-option labeling.

The "board" label used to fall out of `not constraints` being true only for
charuco (which passed constraints=None). Now that charuco calibration also
produces constraints (ConstraintSet.from_charuco), that condition is dead;
labeling is driven by the target type the coordinator hands the presenter.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.workflow_status import StepStatus
from caliscope.gui.presenters.extrinsic_calibration_presenter import (
    CalibrationStepData,
    ExtrinsicCalibrationPresenter,
)


def _make_presenter(extrinsic_target_type=None, image_points_path=None) -> ExtrinsicCalibrationPresenter:
    return ExtrinsicCalibrationPresenter(
        task_manager=MagicMock(),
        camera_array=CameraArray({}),
        image_points_path=image_points_path or Path("does_not_exist.csv"),
        extrinsic_target_type=extrinsic_target_type,
    )


def _write_minimal_image_points(path: Path) -> None:
    """Write a schema-valid image_points.csv with two cameras sharing one point."""
    pd.DataFrame(
        {
            "sync_index": [0, 0],
            "cam_id": [0, 1],
            "object_id": [0, 0],
            "keypoint_id": [0, 0],
            "img_loc_x": [1.0, 2.0],
            "img_loc_y": [3.0, 4.0],
            "obj_loc_x": [0.0, 0.0],
            "obj_loc_y": [0.0, 0.0],
            "obj_loc_z": [0.0, 0.0],
        }
    ).to_csv(path, index=False)


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


def test_refresh_extraction_status_picks_up_late_extraction(qapp, tmp_path):
    """Extraction output written after construction flips the extract step to
    COMPLETE and enables calibration once refresh_extraction_status runs."""
    image_points_path = tmp_path / "image_points.csv"
    presenter = _make_presenter(image_points_path=image_points_path)

    # Nothing on disk yet: extract step is not started, calibration is gated.
    assert presenter.has_extraction_data is False
    steps: list[CalibrationStepData] = []
    presenter.workflow_updated.connect(steps.append)
    presenter.emit_initial_state()
    assert steps[-1].extract[0] == StepStatus.NOT_STARTED

    # Extraction runs on the Multi-Camera tab and writes the CSV.
    _write_minimal_image_points(image_points_path)
    presenter.refresh_extraction_status()

    assert presenter.has_extraction_data is True
    assert steps[-1].extract[0] == StepStatus.COMPLETE


def test_refresh_extraction_status_noop_when_already_loaded(qapp, tmp_path):
    """A second refresh after data is loaded does not re-read or re-emit."""
    image_points_path = tmp_path / "image_points.csv"
    _write_minimal_image_points(image_points_path)
    presenter = _make_presenter(image_points_path=image_points_path)
    assert presenter.has_extraction_data is True

    steps: list[CalibrationStepData] = []
    presenter.workflow_updated.connect(steps.append)
    presenter.refresh_extraction_status()
    assert steps == []
