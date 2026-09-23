"""LensModelDialog reads its video frame off the GUI thread."""

import threading
from pathlib import Path

from PySide6.QtWidgets import QLabel

from caliscope import __root__
from caliscope.cameras.camera_array import CameraArray
from caliscope.gui.widgets import lens_model_dialog
from caliscope.gui.widgets.lens_model_dialog import LensModelDialog
from caliscope.task_manager.task_manager import TaskManager

SESSION = Path(__root__, "tests", "sessions", "4_cam_recording")


def test_first_frame_is_decoded_in_a_worker_thread(qtbot, monkeypatch):
    reader_threads = []
    real_frame_source = lens_model_dialog.FrameSource

    def recording_frame_source(*args, **kwargs):
        reader_threads.append(threading.current_thread())
        return real_frame_source(*args, **kwargs)

    monkeypatch.setattr(lens_model_dialog, "FrameSource", recording_frame_source)
    task_manager = TaskManager()
    cameras = CameraArray.from_toml(SESSION / "camera_array.toml").cameras

    dialog = LensModelDialog(
        cameras=cameras,
        extrinsic_dir=SESSION / "calibration" / "extrinsic",
        task_manager=task_manager,
        initial_cam_id=0,
    )
    qtbot.addWidget(dialog)
    before = dialog.findChild(QLabel, "lensBeforeLabel")
    assert before is not None

    qtbot.waitUntil(lambda: not before.pixmap().isNull(), timeout=10000)
    assert reader_threads
    assert all(thread is not threading.main_thread() for thread in reader_threads)
    task_manager.shutdown()
