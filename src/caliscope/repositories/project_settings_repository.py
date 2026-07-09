"""
Repository for global project configuration settings with in-memory caching.
All persistence errors are converted to ValueError for cleaner API boundaries.
"""

import logging
from pathlib import Path
from typing import Any

from caliscope import persistence

logger = logging.getLogger(__name__)


class ProjectSettingsRepository:
    """
    Persistence gateway for project-wide settings stored in project_settings.toml.

    Settings are loaded once on initialization and cached. Call refresh() to
    reload from disk. This repository is thread-safe as it holds no mutable state
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

    def get_save_xy_points(self) -> bool:
        """Get flag for saving the xy_{tracker}.csv 2D-points artifact (default: True)."""
        return self._cache.get("save_xy_points", True)

    def set_save_xy_points(self, save: bool) -> None:
        """Update xy-points saving flag and persist immediately."""
        settings = self._cache.copy()
        settings["save_xy_points"] = save
        self.save(settings)

    def get_creation_date(self) -> Any:
        """Get project creation date if available."""
        return self._cache.get("creation_date")

    def get_scene_camera_size_multiplier(self) -> float:
        """Get camera frustum size multiplier for 3D visualization (default: 1.0)."""
        return float(self._cache.get("scene_camera_size_multiplier", 1.0))

    def set_scene_camera_size_multiplier(self, multiplier: float) -> None:
        """Update camera frustum size multiplier and persist immediately."""
        settings = self._cache.copy()
        settings["scene_camera_size_multiplier"] = multiplier
        self.save(settings)

    def get_scene_grid_size_multiplier(self) -> float:
        """Get floor grid size multiplier for 3D visualization (default: 1.0)."""
        return float(self._cache.get("scene_grid_size_multiplier", 1.0))

    def set_scene_grid_size_multiplier(self, multiplier: float) -> None:
        """Update floor grid size multiplier and persist immediately."""
        settings = self._cache.copy()
        settings["scene_grid_size_multiplier"] = multiplier
        self.save(settings)

    def get_scene_sphere_size_multiplier(self) -> float:
        """Get point sphere size multiplier for 3D visualization (default: 1.0)."""
        return float(self._cache.get("scene_sphere_size_multiplier", 1.0))

    def set_scene_sphere_size_multiplier(self, multiplier: float) -> None:
        """Update point sphere size multiplier and persist immediately."""
        settings = self._cache.copy()
        settings["scene_sphere_size_multiplier"] = multiplier
        self.save(settings)

    def get_refine_intrinsics(self) -> bool:
        """Get flag for refining intrinsics during extrinsic bundle adjustment (default: True)."""
        return bool(self._cache.get("refine_intrinsics", True))

    def set_refine_intrinsics(self, refine: bool) -> None:
        """Update refine intrinsics flag and persist immediately."""
        settings = self._cache.copy()
        settings["refine_intrinsics"] = refine
        self.save(settings)

    def get_origin_object_id(self) -> int | None:
        """Get selected origin marker id, or None if not set."""
        return self._cache.get("origin_object_id")

    def set_origin_object_id(self, object_id: int | None) -> None:
        """Update selected origin marker id and persist immediately. None removes the setting."""
        settings = self._cache.copy()
        if object_id is None:
            settings.pop("origin_object_id", None)
        else:
            settings["origin_object_id"] = object_id
        self.save(settings)

    def get_origin_sync_index(self) -> int | None:
        """Get selected origin frame's sync index, or None for a static origin marker."""
        return self._cache.get("origin_sync_index")

    def set_origin_sync_index(self, sync_index: int | None) -> None:
        """Update origin frame's sync index and persist immediately. None removes the setting."""
        settings = self._cache.copy()
        if sync_index is None:
            settings.pop("origin_sync_index", None)
        else:
            settings["origin_sync_index"] = sync_index
        self.save(settings)
