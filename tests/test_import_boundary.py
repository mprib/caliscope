"""Verify core modules can be imported without PySide6.

Tests modules that are near the Qt boundary -- either recently cleaned up
or sharing a package with Qt-tainted code. Pure numpy/opencv modules
(alignment, reprojection, triangulation, etc.) are omitted since they have
no plausible path to Qt contamination.
"""

import subprocess
import sys

import pytest


HEADLESS_IMPORTS = [
    # Primary API export -- the highest-value canary
    "caliscope.core.capture_volume",
    # Recently cleaned: Qt imports removed in this branch
    "caliscope.core.charuco",
    # Recently cleaned: Qt made conditional in this branch
    "caliscope.logger",
    # Share a package with Qt-tainted frame_packet_streamer;
    # recording/__init__.py was the contamination vector we fixed
    "caliscope.recording.frame_source",
    "caliscope.recording.video_utils",
    # Package-level __init__.py re-exports -- could pull in Qt transitively
    "caliscope.repositories",
    "caliscope.trackers",
]


@pytest.mark.parametrize("module", HEADLESS_IMPORTS)
def test_import_without_pyside6(module: str) -> None:
    """Each core module must be importable when PySide6 is blocked."""
    script = f"""\
import sys

class PySide6Blocker:
    \"\"\"Meta path finder that makes PySide6 appear uninstalled.\"\"\"
    def find_module(self, fullname, path=None):
        if fullname == "PySide6" or fullname.startswith("PySide6."):
            return self
    def load_module(self, fullname):
        raise ImportError(f"PySide6 is blocked for headless testing: {{fullname}}")

sys.meta_path.insert(0, PySide6Blocker())

import {module}
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Failed to import {module} without PySide6:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
