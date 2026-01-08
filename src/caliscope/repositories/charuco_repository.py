"""
Repository for Charuco calibration board definition.
Simplest repository as Charuco is a single domain object with no dependencies.
"""

import logging
from pathlib import Path

from caliscope.core.charuco import Charuco
from caliscope import persistence

logger = logging.getLogger(__name__)


class CharucoRepository:
    """
    Persistence gateway for Charuco board definition stored in charuco.toml.

    This repository is thread-safe as it holds no state and performs only
    atomic file operations.
    """

    def __init__(self, charuco_path: Path) -> None:
        """
        Args:
            charuco_path: Path to charuco.toml in workspace root
        """
        self.path = charuco_path

    def load(self) -> Charuco:
        """
        Load charuco board definition.

        Returns:
            Charuco instance

        Raises:
            ValueError: If file doesn't exist or contains invalid board parameters
        """
        try:
            charuco = persistence.load_charuco(self.path)
            logger.debug(f"Loaded charuco from {self.path}")
            return charuco
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to load charuco: {e}") from e

    def save(self, charuco: Charuco) -> None:
        """
        Save charuco board definition.

        Args:
            charuco: Charuco to serialize

        Raises:
            ValueError: If save operation fails
        """
        try:
            persistence.save_charuco(charuco, self.path)
            logger.info(f"Saved charuco to {self.path}")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save charuco: {e}") from e
