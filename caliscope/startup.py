# caliscope/startup.py

import logging
from pathlib import Path

import rtoml

from caliscope import APP_SETTINGS_PATH, LOG_FILE_PATH, __package_name__, __repo_url__

logger = logging.getLogger(__name__)


def initialize_app():
    """
    Ensures that necessary application directories and the settings file exist.
    This should be called once when the application starts.
    Returns the loaded user settings.
    """

    """Logs essential application information on startup."""
    logger.info(f"--- Launching {__package_name__} ---")
    logger.info(f"Source code available at: {__repo_url__}")
    logger.info(f"Logs saved to: {LOG_FILE_PATH}")
    logger.info("-------------------------------------------")

    try:
        # If the settings file doesn't exist, create it with defaults
        if not APP_SETTINGS_PATH.exists():
            logger.info(f"Settings file not found. Creating a new one at {APP_SETTINGS_PATH}")
            user_home = Path.home()
            default_settings = {
                "recent_projects": [],
                "last_project_parent": str(user_home),
            }
            with open(APP_SETTINGS_PATH, "w") as f:
                rtoml.dump(default_settings, f)

        # Now, load and return the settings
        logger.info(f"Loading settings from {APP_SETTINGS_PATH}")
        return rtoml.load(APP_SETTINGS_PATH)

    except Exception as e:
        logger.error(f"Failed to initialize application directories or settings: {e}")
        # Depending on how critical this is, you might want to exit or return default settings
        return {}
