"""
Manages global project configuration settings with in-memory caching.
All persistence errors are converted to ValueError for cleaner API boundaries.
"""

import logging
from pathlib import Path
from typing import Any

from caliscope import persistence

logger = logging.getLogger(__name__)


class ProjectSettingsManager:
    """
    Manages project-wide settings stored in project_settings.toml.

    Settings are loaded once on initialization and cached. Call refresh() to
    reload from disk. This manager is thread-safe as it holds no mutable state
    beyond the cached settings dictionary.
    """

    def __init__(self, settings_path: Path) -> None:
        """
        Args:
            settings_path: Path to project_settings.toml in workspace root

        Raises:
            ValueError: If settings file exists but is malformed
        """
        self.path = settings_path
        self._cache: dict[str, Any] = {}
        self.refresh()  # Initial load

    def refresh(self) -> None:
        """Reload settings from disk, updating the cache."""
        try:
            self._cache = persistence.load_project_settings(self.path)
            logger.debug(f"Loaded project settings from {self.path}")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to load project settings: {e}") from e

    def save(self, settings: dict[str, Any]) -> None:
        """
        Save settings to disk and update cache.

        Args:
            settings: Complete settings dictionary. None values are filtered.

        Raises:
            ValueError: If save operation fails
        """
        try:
            persistence.save_project_settings(settings, self.path)
            self._cache = settings.copy()  # Update cache
            logger.debug(f"Saved project settings to {self.path}")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save project settings: {e}") from e

    # Typed accessors provide API stability and prevent magic strings
    def get_camera_count(self) -> int:
        """Get configured camera count (default: 0 for new projects)."""
        return self._cache.get("camera_count", 0)

    def set_camera_count(self, count: int) -> None:
        """Update camera count and persist immediately."""
        settings = self._cache.copy()
        settings["camera_count"] = count
        self.save(settings)

    def get_fps_sync_stream_processing(self) -> int:
        """Get FPS throttle for stream processing (default: 100)."""
        return self._cache.get("fps_sync_stream_processing", 100)

    def set_fps_sync_stream_processing(self, fps: int) -> None:
        """Update FPS throttle and persist immediately."""
        settings = self._cache.copy()
        settings["fps_sync_stream_processing"] = fps
        self.save(settings)

    def get_save_tracked_points_video(self) -> bool:
        """Get flag for saving tracked points overlay video (default: True)."""
        return self._cache.get("save_tracked_points_video", True)

    def set_save_tracked_points_video(self, save: bool) -> None:
        """Update video saving flag and persist immediately."""
        settings = self._cache.copy()
        settings["save_tracked_points_video"] = save
        self.save(settings)

    def get_creation_date(self) -> Any:
        """Get project creation date if available."""
        return self._cache.get("creation_date")
