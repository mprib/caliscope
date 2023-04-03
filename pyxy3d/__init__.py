"""Top-level package for basic_template_repo."""
import os
from pathlib import Path

__package_name__ = "pyxy3d"
__version__ = "v0.0.15"

__author__ = """Mac Prible"""
__email__ = "prible@gmail.com"
__repo_owner_github_user_name__ = "mprib"
__repo_url__ = f"https://github.com/{__repo_owner_github_user_name__}/{__package_name__}/"
__repo_issues_url__ = f"{__repo_url__}issues"


# set up local app data folder and logging
__app_dir__ = Path(os.getenv("LOCALAPPDATA"), __package_name__)
__app_dir__.mkdir(exist_ok=True, parents=True)

__log_dir__ = Path(__app_dir__, "logs")
__log_dir__.mkdir(exist_ok=True, parents=True)

# a helpful reference
__root__ = Path(__file__).parent.parent



print(f"Thank you for using {__package_name__}!")
print(f"This is printing from: {__file__}")
print(f"Source code for this package is available at: {__repo_url__}")
print(f"Data and Log files associated with {__package_name__} are stored in {__app_dir__}")