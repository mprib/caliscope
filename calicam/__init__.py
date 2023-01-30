"""Top-level package for basic_template_repo."""

__package_name__ = "calicam"
__version__ = "v0.0.3"

__author__ = """Mac Prible"""
__email__ = "prible@gmail.com"
__repo_owner_github_user_name__ = "mprib"
__repo_url__ = f"https://github.com/{__repo_owner_github_user_name__}/{__package_name__}/"
__repo_issues_url__ = f"{__repo_url__}issues"


print(f"Thank you for using {__package_name__}!")
print(f"This is printing from: {__file__}")
print(f"Source code for this package is available at: {__repo_url__}")

from calicam.system.default_paths import get_log_file_path
from calicam.system.logging_configuration import configure_logging

configure_logging(log_file_path=get_log_file_path())