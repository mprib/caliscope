"""Repository for ArucoTarget calibration object definition."""

import logging
from pathlib import Path

from caliscope.core.aruco_target import ArucoTarget
from caliscope import persistence

logger = logging.getLogger(__name__)


class ArucoTargetRepository:
    """Persistence gateway for ArucoTarget stored in aruco_target.toml."""

    def __init__(self, aruco_target_path: Path) -> None:
        self.path = aruco_target_path

    def exists(self) -> bool:
        """Check if aruco_target.toml exists."""
        return self.path.exists()

    def load(self) -> ArucoTarget:
        """Load target definition.

        Raises:
            ValueError: If file doesn't exist or contains invalid parameters
        """
        try:
            target = persistence.load_aruco_target(self.path)
            logger.debug(f"Loaded ArucoTarget from {self.path}")
            return target
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to load ArucoTarget: {e}") from e

    def save(self, target: ArucoTarget) -> None:
        """Save target definition.

        Raises:
            ValueError: If save operation fails
        """
        try:
            persistence.save_aruco_target(target, self.path)
            logger.info(f"Saved ArucoTarget to {self.path}")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save ArucoTarget: {e}") from e
