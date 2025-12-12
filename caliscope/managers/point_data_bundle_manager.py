"""
Manager for PointDataBundle persistence with complete snapshotting.

This module provides atomic save/load operations for calibration data bundles,
ensuring data integrity when the active calibration changes. Each bundle is
self-contained, storing its own copy of the camera array used for processing.

Key design principles:
- Short-lived instances: Create, use, discard (no state caching)
- Explicit paths: No knowledge of workspace structure
- Atomic operations: Temp file + rename pattern prevents corruption
- Complete snapshots: Camera array is duplicated in each bundle directory
"""

from pathlib import Path
import rtoml
import logging

from caliscope.calibration.point_data_bundle import PointDataBundle, BundleMetadata
from caliscope.persistence import (
    load_camera_array,
    save_camera_array,
    load_image_points_csv,
    save_image_points_csv,
    load_world_points_csv,
    save_world_points_csv,
    PersistenceError,
)

logger = logging.getLogger(__name__)


class PointDataBundleManager:
    """
    Handles atomic save/load of PointDataBundle with complete snapshots.

    Each bundle directory contains:
    - camera_array.toml: Snapshot of calibration used for processing
    - image_points.csv: 2D observations
    - world_points.csv: 3D triangulated points
    - bundle.toml: Provenance metadata and operations history

    The manager is short-lived and should be created per operation to avoid
    stale state. It delegates all format-specific I/O to the persistence layer.
    """

    def __init__(self, base_path: Path):
        """
        Initialize manager for a specific bundle directory.

        Args:
            base_path: Directory where bundle components will be stored.
                      For calibration: workspace/calibration/extrinsic/CHARUCO
                      For recordings: workspace/recordings/recording_1
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Define file paths relative to base_path
        self.bundle_path = base_path / "bundle.toml"
        self.camera_array_path = base_path / "camera_array.toml"
        self.image_points_path = base_path / "image_points.csv"
        self.world_points_path = base_path / "world_points.csv"

    def load(self) -> PointDataBundle:
        """
        Load complete bundle from self-contained directory.

        Returns:
            PointDataBundle with all components loaded and validated

        Raises:
            PersistenceError: If any required file is missing or corrupted
        """
        try:
            # Load components in dependency order
            camera_array = load_camera_array(self.camera_array_path)
            image_points = load_image_points_csv(self.image_points_path)
            world_points = load_world_points_csv(self.world_points_path)
            metadata = self._load_metadata()

            return PointDataBundle(
                camera_array=camera_array,
                image_points=image_points,
                world_points=world_points,
                metadata=metadata,
            )
        except FileNotFoundError as e:
            raise PersistenceError(
                f"Bundle file missing at {self.base_path}: {e}. "
                f"Expected files: camera_array.toml, image_points.csv, world_points.csv, bundle.toml"
            ) from e
        except Exception as e:
            raise PersistenceError(f"Failed to load bundle from {self.base_path}: {e}") from e

    def save(self, bundle: PointDataBundle) -> None:
        """
        Save all bundle components atomically.

        Writes data files first, then metadata last to mark bundle as complete.
        Uses temp file + rename pattern for atomic metadata write.

        Args:
            bundle: PointDataBundle to persist

        Raises:
            PersistenceError: If any write operation fails
        """
        try:
            # Save components in order: data first, metadata last
            # This ensures bundle.toml only exists if all data is present
            save_camera_array(bundle.camera_array, self.camera_array_path)
            save_image_points_csv(bundle.image_points, self.image_points_path)
            save_world_points_csv(bundle.world_points, self.world_points_path)

            # Save metadata last (marks bundle as complete)
            self._save_metadata(bundle.metadata)

            logger.info(f"Successfully saved PointDataBundle to {self.base_path}")
        except Exception as e:
            raise PersistenceError(f"Failed to save bundle to {self.base_path}: {e}") from e

    def _load_metadata(self) -> BundleMetadata:
        """
        Load metadata from bundle.toml.

        Returns:
            BundleMetadata instance with provenance information

        Raises:
            PersistenceError: If metadata file is missing or malformed
        """
        if not self.bundle_path.exists():
            raise PersistenceError(f"Bundle metadata not found: {self.bundle_path}")

        try:
            data = rtoml.load(self.bundle_path)
            return BundleMetadata(
                created_at=data["created_at"],
                generation_method=data["generation_method"],
                generation_params=data["generation_params"],
                camera_array_path=Path(data["camera_array_path"]),
                operations=data.get("operations", []),
                source_files={k: Path(v) for k, v in data.get("source_files", {}).items()},
            )
        except Exception as e:
            raise PersistenceError(f"Failed to load metadata from {self.bundle_path}: {e}") from e

    def _save_metadata(self, metadata: BundleMetadata) -> None:
        """
        Save metadata to bundle.toml with atomic write.

        Writes to temporary file first, then renames to ensure atomicity.
        This prevents corruption if write is interrupted.

        Args:
            metadata: BundleMetadata to serialize
        """
        # Prepare data for TOML serialization
        data = {
            "created_at": metadata.created_at,
            "generation_method": metadata.generation_method,
            "generation_params": metadata.generation_params,
            "camera_array_path": str(metadata.camera_array_path),
            "operations": metadata.operations,
            "source_files": {k: str(v) for k, v in metadata.source_files.items()},
        }

        # Atomic write: temp file + rename
        temp_path = self.bundle_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                rtoml.dump(data, f)
            temp_path.rename(self.bundle_path)
        except Exception as e:
            # Clean up temp file if write fails
            if temp_path.exists():
                temp_path.unlink()
            raise PersistenceError(f"Failed to save metadata to {self.bundle_path}: {e}") from e
