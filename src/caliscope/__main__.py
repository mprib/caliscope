from __future__ import annotations

import faulthandler
import os
import sys
import tempfile

# Write faulthandler trace to a file that survives the segfault
# (pipe buffers don't flush on SIGSEGV)
_faulthandler_file = open(  # noqa: SIM115
    os.path.join(tempfile.gettempdir(), "caliscope_faulthandler.log"), "w"
)
faulthandler.enable(file=_faulthandler_file, all_threads=True)

from caliscope import MODELS_DIR  # noqa: E402


def CLI_parser():
    # Qt env vars must be set before any Qt imports
    os.environ.setdefault("QT_API", "pyside6")

    try:
        import PySide6
    except ImportError:
        print(
            "The caliscope GUI requires additional dependencies.\nInstall with: pip install caliscope[gui]",
            file=sys.stderr,
        )
        sys.exit(1)

    # pyside6-essentials compatibility: qtpy needs PySide6.__version__ which essentials doesn't provide
    # PySide6.QtCore.__version__ exists at runtime (C++ binding) but PySide6's type stubs omit it.
    from PySide6.QtCore import __version__ as _qt_version  # type: ignore[attr-defined]

    PySide6.__version__ = _qt_version

    from caliscope.gui.main_widget import launch_main
    from caliscope.logger import setup_logging
    from caliscope.startup import initialize_app
    from caliscope.trackers import tracker_registry

    setup_logging()
    initialize_app()

    tracker_registry.scan_onnx_models(MODELS_DIR)

    import argparse

    parser = argparse.ArgumentParser(prog="caliscope")
    parser.add_argument("--workspace", type=str, help="Path to workspace directory to open on launch")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    launch_main(workspace=args.workspace)
