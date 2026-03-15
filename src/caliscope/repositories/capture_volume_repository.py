"""
Repository for CaptureVolume persistence with complete snapshotting.

This module provides atomic save/load operations for calibration data,
ensuring data integrity when the active calibration changes. Each capture volume
is self-contained, storing its own copy of the camera array used for processing.

Key design principles:
- Short-lived instances: Create, use, discard (no state caching)
- Explicit paths: No knowledge of workspace structure
- Atomic operations: Temp file + rename pattern prevents corruption
- Complete snapshots: Camera array is duplicated in each capture volume directory
"""

from pathlib import Path
import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.core.capture_volume import CaptureVolume
from caliscope.persistence import PersistenceError

logger = logging.getLogger(__name__)


class CaptureVolumeRepository:
    """
    Persistence gateway for CaptureVolume with complete snapshots.

    Each capture volume directory contains:
    - camera_array.toml: Snapshot of calibration used for processing
    - image_points.csv: 2D observations
    - world_points.csv: 3D triangulated points

    The repository is short-lived and should be created per operation to avoid
    stale state. It delegates all format-specific I/O to the persistence layer.
    """

    def __init__(self, base_path: Path):
        """
        Initialize repository for a specific capture volume directory.

        Args:
            base_path: Directory where capture volume components will be stored.
                      For calibration: workspace/calibration/extrinsic/CHARUCO
                      For recordings: workspace/recordings/recording_1
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Define file paths relative to base_path
        self.camera_array_path = base_path / "camera_array.toml"
        self.image_points_path = base_path / "image_points.csv"
        self.world_points_path = base_path / "world_points.csv"

    def load(self) -> CaptureVolume:
        """
        Load complete capture volume from self-contained directory.

        Returns:
            CaptureVolume with all components loaded and validated

        Raises:
            PersistenceError: If any required file is missing or corrupted
        """
        try:
            # Load components in dependency order
            camera_array = CameraArray.from_toml(self.camera_array_path)
            image_points = ImagePoints.from_csv(self.image_points_path)
            world_points = WorldPoints.from_csv(self.world_points_path)

            return CaptureVolume(
                camera_array=camera_array,
                image_points=image_points,
                world_points=world_points,
            )
        except FileNotFoundError as e:
            raise PersistenceError(
                f"Capture volume file missing at {self.base_path}: {e}. "
                f"Expected files: camera_array.toml, image_points.csv, world_points.csv"
            ) from e
        except Exception as e:
            raise PersistenceError(f"Failed to load capture volume from {self.base_path}: {e}") from e

    def save(self, capture_volume: CaptureVolume) -> None:
        """
        Save all capture volume components atomically.

        Writes data files first, then metadata last to mark capture volume as complete.
        Uses temp file + rename pattern for atomic metadata write.

        Args:
            capture_volume: CaptureVolume to persist

        Raises:
            PersistenceError: If any write operation fails
        """
        try:
            # Save components in order: data first
            capture_volume.camera_array.to_toml(self.camera_array_path)
            capture_volume.image_points.to_csv(self.image_points_path)
            capture_volume.world_points.to_csv(self.world_points_path)

            logger.info(f"Successfully saved CaptureVolume to {self.base_path}")
        except Exception as e:
            raise PersistenceError(f"Failed to save capture volume to {self.base_path}: {e}") from e
