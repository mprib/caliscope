"""
Repository for Chessboard calibration pattern definition.
Simplest repository as Chessboard is a single frozen dataclass with no dependencies.
"""

import logging
from pathlib import Path

from caliscope.core.chessboard import Chessboard
from caliscope import persistence

logger = logging.getLogger(__name__)


class ChessboardRepository:
    """
    Persistence gateway for Chessboard pattern definition stored in chessboard.toml.

    This repository is thread-safe as it holds no state and performs only
    atomic file operations.
    """

    def __init__(self, chessboard_path: Path) -> None:
        """
        Args:
            chessboard_path: Path to chessboard.toml in calibration directory
        """
        self.path = chessboard_path

    def exists(self) -> bool:
        """Check if chessboard.toml exists."""
        return self.path.exists()

    def load(self) -> Chessboard:
        """
        Load chessboard pattern definition.

        Returns:
            Chessboard instance

        Raises:
            ValueError: If file doesn't exist or contains invalid parameters
        """
        try:
            chessboard = persistence.load_chessboard(self.path)
            logger.debug(f"Loaded chessboard from {self.path}")
            return chessboard
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to load chessboard: {e}") from e

    def save(self, chessboard: Chessboard) -> None:
        """
        Save chessboard pattern definition.

        Args:
            chessboard: Chessboard to serialize

        Raises:
            ValueError: If save operation fails
        """
        try:
            persistence.save_chessboard(chessboard, self.path)
            logger.info(f"Saved chessboard to {self.path}")
        except persistence.PersistenceError as e:
            raise ValueError(f"Failed to save chessboard: {e}") from e
