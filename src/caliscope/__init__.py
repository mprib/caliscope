"""Top-level package for Caliscope."""

# VTK/PyVista environment configuration
# Must be set before any Qt or VTK imports
import os
import sys

if sys.platform == "linux" and os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

os.environ.setdefault("QT_API", "pyside6")

# PySide6-essentials compatibility shim for pyqtgraph
# pyqtgraph expects PySide6.__version__ and PySide6.__version_info__ at the top level,
# but pyside6-essentials only exposes these in PySide6.QtCore.
# This shim adds the missing attributes before pyqtgraph is imported.
# Can be removed once pyqtgraph fixes: https://github.com/pyqtgraph/pyqtgraph/issues/2048
import PySide6
from PySide6 import QtCore

# Monkey-patch for pyqtgraph compatibility - stubs don't know about this
PySide6.__version__ = QtCore.__version__  # type: ignore[attr-defined]
PySide6.__version_info__ = QtCore.__version_info__  # type: ignore[attr-defined]

from pathlib import Path  # noqa: E402

from platformdirs import user_data_dir  # noqa: E402

__package_name__ = "caliscope"
__author__ = "Mac Prible"
__email__ = "prible@utexas.edu"
__repo_owner_github_user_name__ = "mprib"
__repo_url__ = f"https://github.com/{__repo_owner_github_user_name__}/{__package_name__}/"
__repo_issues_url__ = f"{__repo_url__}issues"

# --- Use platformdirs to define standard paths ---
# - Windows: C:\Users\<user>\AppData\Local\Mac Prible\caliscope
# - macOS:   ~/Library/Application Support/caliscope
# - Linux:   ~/.local/share/caliscope
APP_DIR = Path(user_data_dir(appname=__package_name__))

# user_log_dir will be:
# - Windows: C:\Users\<user>\AppData\Local\Mac Prible\caliscope\Logs
# - macOS:   ~/Library/Logs/caliscope
# - Linux:   ~/.config/caliscope/logs  (or ~/.local/state/... depending on XDG spec)
LOG_DIR = APP_DIR / "logs"
LOG_FILE_PATH = LOG_DIR / "caliscope.log"

# Define the path to the settings file
APP_SETTINGS_PATH = APP_DIR / "settings.toml"

# A helpful reference to the source code root
__root__ = Path(__file__).parent.parent.parent
