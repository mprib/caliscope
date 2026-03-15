from __future__ import annotations

import os
import sys
from pathlib import Path

from caliscope import MODELS_DIR


def _seed_default_model_cards(models_dir: "Path") -> None:
    """Copy shipped model card templates on first run.

    Trigger: MODELS_DIR does not exist yet. If it already exists (even if
    empty), the user owns that directory and we do not touch it.
    """
    import importlib.resources
    import logging

    logger = logging.getLogger(__name__)

    if models_dir.exists():
        return

    models_dir.mkdir(parents=True, exist_ok=True)

    package_cards = importlib.resources.files("caliscope.trackers.model_cards")
    for resource in package_cards.iterdir():
        if resource.name.endswith(".toml"):
            try:
                dest = models_dir / resource.name
                dest.write_text(resource.read_text())
                logger.info("Seeded default model card: %s", resource.name)
            except OSError:
                logger.warning("Failed to seed model card: %s", resource.name, exc_info=True)


def CLI_parser():
    # Qt env vars must be set before any Qt or VTK imports
    # Linux + Wayland: VTK doesn't support native Wayland rendering, force XWayland
    if sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

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
    from PySide6.QtCore import __version__ as _qt_version

    PySide6.__version__ = _qt_version

    from caliscope.gui.main_widget import launch_main
    from caliscope.logger import setup_logging
    from caliscope.startup import initialize_app
    from caliscope.trackers import tracker_registry

    setup_logging()
    initialize_app()

    # Seed default model cards on first run, then scan
    _seed_default_model_cards(MODELS_DIR)
    tracker_registry.scan_onnx_models(MODELS_DIR)

    if len(sys.argv) == 1:
        launch_main()
