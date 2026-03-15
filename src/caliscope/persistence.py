"""Low-level I/O utilities for safe file writes.

Domain objects own their serialization format (to_toml/from_toml, to_csv/from_csv).
This module provides only:
- Atomic write helpers (temp file + fsync + rename)
- Shared constants
- Generic dict-based TOML load/save for untyped config files
"""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import rtoml

logger = logging.getLogger(__name__)


class PersistenceError(Exception):
    """Raised when file I/O or data validation fails."""

    pass


CSV_FLOAT_PRECISION = "%.6f"  # 6 decimal places = micron precision at meter scale


def _safe_write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write CSV via temp file with fsync to prevent data loss on crash.

    NTFS journals metadata but not data by default. A crash between file
    allocation and data flush produces null bytes in the output. Writing to
    a temp file, fsyncing, then atomically renaming avoids this.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        df.to_csv(f, **kwargs)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _safe_write_toml(data: dict, path: Path) -> None:
    """Write TOML via temp file with fsync to prevent data loss on crash."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        rtoml.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# --- Generic dict-based config helpers (no domain object involved) ---


def load_capture_volume_metadata(path: Path) -> dict[str, Any]:
    """Load capture volume metadata from TOML file.

    Metadata includes: stage (optimization stage), origin_sync_index
    (frame index where origin was set), and other capture volume configuration.

    Raises:
        PersistenceError: If file doesn't exist or format is invalid
    """
    if not path.exists():
        raise PersistenceError(f"Capture volume metadata file not found: {path}")

    try:
        data = rtoml.load(path)
        # Handle missing keys gracefully - return None if not present
        return {
            "stage": data.get("stage"),
            "origin_sync_index": data.get("origin_sync_index"),
        }
    except Exception as e:
        raise PersistenceError(f"Failed to load capture volume metadata from {path}: {e}") from e


def save_capture_volume_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Save capture volume metadata to TOML file.

    Raises:
        PersistenceError: If serialization or write fails
    """
    try:
        # Filter out None values - TOML doesn't have null type
        data_to_save = {k: v for k, v in metadata.items() if v is not None}
        _safe_write_toml(data_to_save, path)
    except Exception as e:
        raise PersistenceError(f"Failed to save capture volume metadata to {path}: {e}") from e


def load_project_settings(path: Path) -> dict[str, Any]:
    """Load project settings from TOML file.

    Returns empty dict if file doesn't exist (backward compat with new projects).

    Raises:
        PersistenceError: If file exists but format is invalid
    """
    if not path.exists():
        # Return empty dict for new projects
        return {}

    try:
        data = rtoml.load(path)
        return data
    except Exception as e:
        raise PersistenceError(f"Failed to load project settings from {path}: {e}") from e


def save_project_settings(settings: dict[str, Any], path: Path) -> None:
    """Save project settings to TOML file.

    Raises:
        PersistenceError: If serialization or write fails
    """
    try:
        # Filter out None values
        data_to_save = {k: v for k, v in settings.items() if v is not None}
        _safe_write_toml(data_to_save, path)
    except Exception as e:
        raise PersistenceError(f"Failed to save project settings to {path}: {e}") from e
