"""Smoke test: verify the entry point module loads on all platforms.

The __main__ module runs code at import time (faulthandler setup, path
resolution). A hardcoded Unix path once crashed the app on Windows.
This test catches that class of platform-specific startup failure.
"""

from __future__ import annotations

import subprocess
import sys


def test_entry_point_module_imports():
    result = subprocess.run(
        [sys.executable, "-c", "import caliscope.__main__"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Entry point module failed to import:\n{result.stderr}"


if __name__ == "__main__":
    test_entry_point_module_imports()
    print("Smoke test passed.")
