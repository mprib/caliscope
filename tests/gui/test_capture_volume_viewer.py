"""Focused contracts for the standalone capture-volume viewer and playback timeline."""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import cast

import pandas as pd
import pytest
from PySide6.QtWidgets import QTabWidget, QWidget

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import WorldPoints
from caliscope.core.capture_volume import CaptureVolume
from caliscope.gui import view_capture_volume
from caliscope.gui.view_models.playback_view_model import PlaybackViewModel
from caliscope.gui.widgets.qt3d_playback_widget import Qt3DPlaybackWidget


def _view_model(indices: list[int]) -> PlaybackViewModel:
    world_points = WorldPoints(
        pd.DataFrame(
            [
                {
                    "sync_index": index,
                    "object_id": 0,
                    "keypoint_id": 0,
                    "x_coord": float(index),
                    "y_coord": 0.0,
                    "z_coord": 1.0,
                    "frame_time": float(index),
                }
                for index in indices
            ]
        )
    )
    return PlaybackViewModel(CameraArray({}), world_points, fps=30)


def test_gui_package_import_does_not_import_qt() -> None:
    """The public convenience API remains importable in non-GUI workflows."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import caliscope.gui; assert 'PySide6.QtWidgets' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_view_capture_volume_rejects_invalid_ownership(qapp) -> None:  # noqa: ARG001
    """The blocking helper does not attach itself to an application's event loop."""
    with pytest.raises(RuntimeError, match="owns its QApplication"):
        view_capture_volume(cast(CaptureVolume, object()))


def test_view_capture_volume_requires_the_main_thread() -> None:
    """Qt application creation is confined to Python's main thread."""
    errors: list[Exception] = []

    def call_viewer() -> None:
        try:
            view_capture_volume(cast(CaptureVolume, object()))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=call_viewer)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert str(errors[0]) == "view_capture_volume() must be called from the main thread."


def test_sparse_sync_indices_drive_internal_slider_positions(qapp) -> None:  # noqa: ARG001
    """Playback visits every available sparse frame while public updates use real indices."""
    widget = Qt3DPlaybackWidget(_view_model([0, 3, 6]))

    assert (widget.slider.minimum(), widget.slider.maximum(), widget.slider.value()) == (0, 2, 0)

    widget._advance_frame()
    assert widget.sync_index == 3
    widget._advance_frame()
    assert widget.sync_index == 6
    widget._advance_frame()
    assert widget.sync_index == 0

    widget.set_sync_index(4)
    assert widget.sync_index == 4
    assert widget.slider.value() == 0


def test_view_model_replacement_preserves_exact_sync_or_uses_first_available(qapp) -> None:  # noqa: ARG001
    """Sparse timeline replacement cannot silently select a numerical neighbor."""
    widget = Qt3DPlaybackWidget(_view_model([0, 3, 6]))
    widget.set_sync_index(3)

    widget.set_view_model(_view_model([0, 3, 9]), preserve_camera=True)
    assert widget.sync_index == 3
    assert widget.slider.value() == 1

    widget.set_view_model(_view_model([2, 3, 8]))
    assert widget.sync_index == 2
    assert widget.slider.value() == 0


def test_switching_away_from_the_tab_stops_playback(qapp) -> None:  # noqa: ARG001
    """A hidden widget left playing keeps forcing redraws, so hiding stops playback."""
    tabs = QTabWidget()
    widget = Qt3DPlaybackWidget(_view_model([0, 3, 6]))
    tabs.addTab(widget, "3D")
    tabs.addTab(QWidget(), "other")
    tabs.show()
    widget.play_button.click()
    assert widget.playback_timer.isActive()

    tabs.setCurrentIndex(1)

    assert not widget.is_playing
    assert not widget.playback_timer.isActive()
    tabs.close()


@pytest.mark.parametrize("indices", [[], [-1], [0]])
def test_non_playable_timelines_disable_playback_controls(qapp, indices: list[int]) -> None:  # noqa: ARG001
    """Camera-only, static-only, and single-frame views cannot start playback."""
    view_model = PlaybackViewModel.from_camera_array_only(CameraArray({})) if not indices else _view_model(indices)
    widget = Qt3DPlaybackWidget(view_model)

    assert not widget.play_button.isEnabled()
    assert not widget.slider.isEnabled()
    widget._toggle_playback(True)
    assert not widget.is_playing
    assert not widget.playback_timer.isActive()
