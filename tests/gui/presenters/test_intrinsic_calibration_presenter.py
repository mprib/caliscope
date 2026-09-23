"""Tests for IntrinsicCalibrationPresenter state handling."""

from pathlib import Path

import pytest

from caliscope import __root__
from caliscope.cameras.camera_array import CameraData
from caliscope.core.charuco import Charuco
from caliscope.gui.presenters.intrinsic_calibration_presenter import (
    IntrinsicCalibrationPresenter,
    IntrinsicCalibrationState,
)
from caliscope.trackers.charuco_tracker import CharucoTracker

SESSION = Path(__root__, "tests", "sessions", "prerecorded_calibration")


@pytest.fixture
def presenter(fake_task_manager):
    presenter = IntrinsicCalibrationPresenter(
        camera=CameraData(cam_id=0, size=(1280, 720)),
        video_path=SESSION / "calibration" / "intrinsic" / "cam_0.mp4",
        tracker=CharucoTracker(Charuco.from_toml(SESSION / "charuco.toml")),
        task_manager=fake_task_manager,
        frame_skip=4,
    )
    yield presenter
    presenter.cleanup()


def test_pending_calibration_task_blocks_restart(presenter, fake_task_manager, qtbot):
    """A submitted calibration that has not started yet still counts as calibrating."""
    presenter.start_calibration()
    qtbot.waitUntil(lambda: bool(fake_task_manager.handles("Intrinsic calibration cam_id 0")), timeout=30000)

    assert presenter.state == IntrinsicCalibrationState.CALIBRATING

    presenter.start_calibration()

    assert presenter.state == IntrinsicCalibrationState.CALIBRATING
