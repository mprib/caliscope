"""
Manages camera array data (intrinsics, extrinsics, metadata).
Provides atomic operations for individual camera updates.
"""

import logging
from pathlib import Path

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope import persistence

logger = logging.getLogger(__name__)


class CameraArrayManager:
    """
    Manages camera array data stored in camera_array.toml.

    This manager is thread-safe as it holds no mutable state and each operation
    is independent. The save_camera() method uses a load-modify-save pattern
    which is correct but not optimized for high-frequency updates.
    """

    def __init__(self, camera_array_path: Path) -> None:
        """
        Args:
            camera_array_path: Path to camera_array.toml in workspace root
        """
        self.path = camera_array_path

    def load(self) -> CameraArray:
        """
        Load complete camera array.

        Returns:
            CameraArray instance. Returns empty array if file doesn't exist.

        Raises:
            ValueError: If file exists but contains malformed data
        """
        try:
            return persistence.load_camera_array(self.path)
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to load camera array: {e}") from e

    def save(self, camera_array: CameraArray) -> None:
        """
        Save complete camera array.

        Args:
            camera_array: CameraArray to serialize

        Raises:
            ValueError: If save operation fails
        """
        try:
            persistence.save_camera_array(camera_array, self.path)
            logger.info(f"Saved camera array with {len(camera_array.cameras)} cameras")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save camera array: {e}") from e

    def save_camera(self, camera: CameraData) -> None:
        """
        Save a single camera by loading, updating, and saving the full array.

        This is an atomic operation but inefficient for bulk updates. If profiling
        shows this as a bottleneck, consider adding a dedicated persistence
        function for partial updates.

        Args:
            camera: CameraData to save/update

        Raises:
            ValueError: If load or save operation fails
        """
        try:
            camera_array = self.load()
            camera_array.cameras[camera.port] = camera
            self.save(camera_array)
            logger.debug(f"Updated camera {camera.port} in array")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save camera {camera.port}: {e}") from e
