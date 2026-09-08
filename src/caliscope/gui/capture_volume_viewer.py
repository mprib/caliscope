"""Standalone Qt3D viewer for a calibrated capture volume."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from caliscope.core.capture_volume import CaptureVolume
    from caliscope.tracker import WireFrameView


def view_capture_volume(
    volume: CaptureVolume,
    *,
    wireframe: WireFrameView | None = None,
    fps: int = 30,
) -> None:
    """Open a blocking 3D viewer for a calibrated capture volume.

    This convenience API owns a standalone QApplication. Call it only from a
    process without an existing Qt application; embedded applications should
    construct :class:`Qt3DPlaybackWidget` themselves.
    """
    if fps <= 0:
        raise ValueError("fps must be a positive viewing cadence")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("view_capture_volume() must be called from the main thread.")

    try:
        from PySide6.QtWidgets import QApplication, QMainWindow
    except ImportError as error:
        raise RuntimeError(
            "The capture-volume viewer requires the GUI dependencies. "
            "Install Caliscope with `pip install caliscope[gui]`."
        ) from error

    if QApplication.instance() is not None:
        raise RuntimeError(
            "view_capture_volume() owns its QApplication and cannot run inside "
            "an existing Qt application. Create Qt3DPlaybackWidget directly instead."
        )

    app = QApplication([])
    gc_timer = None
    disable_gc = None
    try:
        try:
            from caliscope.gui.gc_confinement import disable as disable_gc, enable
            from caliscope.gui.geometry.wireframe import wireframe_segments_from_view
            from caliscope.gui.view_models.playback_view_model import PlaybackViewModel
            from caliscope.gui.widgets.qt3d_playback_widget import Qt3DPlaybackWidget, opengl_available
        except ImportError as error:
            raise RuntimeError(
                "The capture-volume viewer requires the GUI dependencies. "
                "Install Caliscope with `pip install caliscope[gui]`."
            ) from error

        if not opengl_available():
            raise RuntimeError(
                "The capture-volume viewer could not create an OpenGL context. "
                "Install a working graphics driver or retry with LIBGL_ALWAYS_SOFTWARE=1."
            )

        gc_timer = enable()
        segments = wireframe_segments_from_view(wireframe) if wireframe is not None else []
        view_model = PlaybackViewModel(
            camera_array=volume.camera_array,
            world_points=volume.world_points,
            wireframe_segments=segments,
            fps=fps,
        )
        window = QMainWindow()
        window.setWindowTitle("Caliscope Capture Volume")
        widget = Qt3DPlaybackWidget(view_model)
        window.setCentralWidget(widget)
        window.resize(1200, 800)
        window.show()
        app.exec()
    finally:
        if gc_timer is not None:
            assert disable_gc is not None
            disable_gc(gc_timer)
