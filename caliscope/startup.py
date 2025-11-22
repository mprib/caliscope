# caliscope/startup.py

import logging
import os  # <--- Added to set environment variables
from pathlib import Path

import rtoml

from caliscope import APP_SETTINGS_PATH, LOG_FILE_PATH, __package_name__, __repo_url__

logger = logging.getLogger(__name__)


def initialize_app():
    """
    Ensures that necessary application directories and the settings file exist.
    Sets environment variables based on those settings (e.g. Software Rendering).

    WARNING: This function must be called BEFORE any PySide6/Qt imports in the main script.
    """

    # Setup basic logging for startup
    logger.info(f"--- Launching {__package_name__} ---")
    logger.info(f"Source code available at: {__repo_url__}")
    logger.info(f"Logs saved to: {LOG_FILE_PATH}")
    logger.info("-------------------------------------------")

    try:
        # 1. If the settings file doesn't exist, create it with defaults
        if not APP_SETTINGS_PATH.exists():
            logger.info(f"Settings file not found. Creating a new one at {APP_SETTINGS_PATH}")
            user_home = Path.home()

            default_settings = {
                "recent_projects": [],
                "last_project_parent": str(user_home),
                "force_cpu_rendering": False,
            }

            with open(APP_SETTINGS_PATH, "w") as f:
                rtoml.dump(default_settings, f)

        # 2. Load the settings
        logger.info(f"Loading settings from {APP_SETTINGS_PATH}")
        settings = rtoml.load(APP_SETTINGS_PATH)

        # 3. Apply Environment Variable Logic
        # This must happen now, before the main script imports PySide6
        if settings.get("force_cpu_rendering", False):
            logger.warning("Force CPU Rendering enabled in settings. Setting LIBGL_ALWAYS_SOFTWARE=1")
            os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

        return settings

    except Exception as e:
        logger.error(f"Failed to initialize application directories or settings: {e}")
        return {}
