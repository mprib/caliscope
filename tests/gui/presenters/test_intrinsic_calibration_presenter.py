"""Tests for IntrinsicCalibrationPresenter state handling."""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QLabel

from caliscope import __root__
from caliscope.cameras.camera_array import CameraData
from caliscope.core.charuco import Charuco
from caliscope.gui.presenters.intrinsic_calibration_presenter import (
    IntrinsicCalibrationPresenter,
    IntrinsicCalibrationState,
)
from caliscope.gui.views.intrinsic_calibration_widget import IntrinsicCalibrationWidget
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


def test_calibration_failure_is_shown_in_widget(presenter, fake_task_manager, qtbot, qapp):
    """A failed calibration task reaches the widget and returns the presenter to READY."""
    widget = IntrinsicCalibrationWidget(presenter)
    qtbot.addWidget(widget)

    presenter.start_calibration()
    qtbot.waitUntil(lambda: bool(fake_task_manager.handles("Intrinsic calibration cam_id 0")), timeout=30000)
    (handle,) = fake_task_manager.handles("Intrinsic calibration cam_id 0")

    fake_task_manager.fail(handle, "RuntimeError", "calibrateCamera diverged")
    qapp.processEvents()

    assert presenter.state == IntrinsicCalibrationState.READY
    error_label = widget.findChild(QLabel, "calibrationErrorLabel")
    assert error_label is not None
    assert not error_label.isHidden()
    assert "calibrateCamera diverged" in error_label.text()
    widget.close()  # stops the render thread
