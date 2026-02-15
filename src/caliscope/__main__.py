# Platform compatibility patches (must run before Qt imports)
import os
import sys

# Linux + Wayland: VTK doesn't support native Wayland rendering, force XWayland
if sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# pyside6-essentials compatibility: qtpy needs PySide6.__version__ which essentials doesn't provide
import PySide6
from PySide6.QtCore import __version__ as _qt_version

PySide6.__version__ = _qt_version

import logging  # noqa: E402

from caliscope import MODELS_DIR  # noqa: E402
from caliscope.gui.main_widget import launch_main  # noqa: E402
from caliscope.logger import setup_logging  # noqa: E402
from caliscope.startup import initialize_app  # noqa: E402
from caliscope.trackers import tracker_registry  # noqa: E402

setup_logging()
initialize_app()
logger = logging.getLogger(__name__)

# Scan for ONNX models at app startup
MODELS_DIR.mkdir(parents=True, exist_ok=True)
tracker_registry.scan_onnx_models(MODELS_DIR)


def CLI_parser():
    if len(sys.argv) == 1:
        launch_main()

    if len(sys.argv) == 2:
        sys.argv[1]
