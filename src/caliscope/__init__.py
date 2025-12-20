"""Top-level package for Caliscope."""

from pathlib import Path

from platformdirs import user_data_dir

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
