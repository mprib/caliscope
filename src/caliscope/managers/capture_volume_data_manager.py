"""
Manages capture volume data: point estimates (2D-3D correspondences) and
capture volume metadata (stage, origin index). These are combined because
they represent different facets of the same domain concept and are always
loaded/saved together during calibration workflows.
"""

import logging
from pathlib import Path
from typing import Any

from caliscope.calibration.capture_volume.capture_volume import CaptureVolume
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope import persistence

logger = logging.getLogger(__name__)


class CaptureVolumeDataManager:
    """
    Manages capture volume data stored across two files:
    - point_estimates.toml: 2D-3D point correspondences
    - capture_volume.toml: Metadata (stage, origin_sync_index)

    This manager is thread-safe as it holds no mutable state. The
    save_capture_volume() method coordinates multiple persistence calls
    which is acceptable for this domain; if this logic grows, consider
    extracting a dedicated use-case service.
    """

    def __init__(self, workspace_path: Path) -> None:
        """
        Args:
            workspace_path: Root workspace directory (not a file path)
        """
        self.workspace_path = workspace_path
        self.point_estimates_path = workspace_path / "point_estimates.toml"
        self.metadata_path = workspace_path / "capture_volume.toml"

    def load_point_estimates(self) -> PointEstimates:
        """
        Load 2D-3D point correspondences.

        Returns:
            PointEstimates instance

        Raises:
            ValueError: If file doesn't exist or contains malformed data
        """
        try:
            return persistence.load_point_estimates(self.point_estimates_path)
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to load point estimates: {e}") from e

    def save_point_estimates(self, point_estimates: PointEstimates) -> None:
        """
        Save 2D-3D point correspondences.

        Args:
            point_estimates: PointEstimates to serialize

        Raises:
            ValueError: If save operation fails
        """
        try:
            persistence.save_point_estimates(point_estimates, self.point_estimates_path)
            logger.debug(f"Saved point estimates to {self.point_estimates_path}")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save point estimates: {e}") from e

    def load_metadata(self) -> dict[str, Any]:
        """
        Load capture volume metadata.

        Returns:
            Dictionary with keys: stage, origin_sync_index. Empty dict if file
            doesn't exist (for backward compatibility).

        Raises:
            ValueError: If file exists but format is invalid
        """
        try:
            return persistence.load_capture_volume_metadata(self.metadata_path)
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to load capture volume metadata: {e}") from e

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Save capture volume metadata.

        Args:
            metadata: Dictionary with keys: stage, origin_sync_index

        Raises:
            ValueError: If save operation fails
        """
        try:
            persistence.save_capture_volume_metadata(metadata, self.metadata_path)
            logger.debug(f"Saved capture volume metadata to {self.metadata_path}")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save capture volume metadata: {e}") from e

    def save_capture_volume(self, capture_volume: CaptureVolume) -> None:
        """
        Convenience method to save all capture volume data atomically.

        Args:
            capture_volume: CaptureVolume instance to serialize

        Raises:
            ValueError: If any save operation fails
        """
        try:
            self.save_point_estimates(capture_volume.point_estimates)
            metadata = {
                "stage": capture_volume.stage,
                "origin_sync_index": capture_volume.origin_sync_index,
            }
            self.save_metadata(metadata)
            logger.info("Saved complete capture volume data")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save capture volume: {e}") from e

    def exists(self) -> bool:
        """
        Check if capture volume data exists (both files present).

        Returns:
            True if both point estimates and metadata files exist
        """
        return self.point_estimates_path.exists() and self.metadata_path.exists()
