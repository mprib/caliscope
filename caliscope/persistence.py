# --- File: caliscope/persistence.py ---

"""
Persistence layer for Caliscope project state.

This module provides all file I/O operations for domain objects, using a
file-per-object approach. Each domain object (CameraArray, Charuco, etc.)
has its own TOML file in the workspace root for atomic updates and clear
separation of concerns.

Supported formats:
- TOML: For structured domain objects (CameraArray, Charuco, PointEstimates, etc.)
- CSV: For tabular point data (image points, world points)

All functions accept pathlib.Path objects and raise PersistenceError for any
I/O or validation failures.
"""

import logging
from pathlib import Path
from typing import Any
import cv2

import numpy as np
import pandas as pd
import rtoml

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.calibration.charuco import Charuco
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.array_initialization.paired_pose_network import PairedPoseNetwork

logger = logging.getLogger(__name__)


class PersistenceError(Exception):
    """Raised when file I/O or data validation fails."""

    pass


# ============================================================================
# Module Constants
# ============================================================================

CSV_FLOAT_PRECISION = "%.6f"  # 6 decimal places = micron precision at meter scale


# ============================================================================
# Private Helper Functions
# ============================================================================


def _array_to_list(arr: np.ndarray | None) -> list | None:
    """
    Convert numpy array to list for TOML serialization.

    Args:
        arr: Numpy array or None

    Returns:
        List representation or None
    """
    return arr.tolist() if arr is not None else None


def _list_to_array(lst: Any, dtype=np.float64) -> np.ndarray | None:
    """
    Convert list back to numpy array from TOML deserialization.

    Handles both proper TOML null (None) and string literal "null".

    Args:
        lst: List representation, None, "null" string, or other value from TOML
        dtype: Numpy dtype for array reconstruction

    Returns:
        Numpy array or None

    Raises:
        ValueError: If lst is not None, "null", or a valid list
    """
    if lst is None or lst == "null":
        return None
    if not isinstance(lst, list):
        raise ValueError(f"Expected list or None, got {type(lst).__name__}: {lst}")
    return np.array(lst, dtype=dtype)


def _write_toml(data: dict, path: Path) -> None:
    """
    Write TOML file directly with error handling.

    Args:
        data: Dictionary to serialize
        path: Target file path

    Raises:
        PersistenceError: If write fails
    """
    try:
        with open(path, "w") as f:
            rtoml.dump(data, f)
    except Exception as e:
        raise PersistenceError(f"Failed to write {path}: {e}") from e


# ============================================================================
# Core Domain Objects
# ============================================================================


def load_camera_array(path: Path) -> CameraArray:
    """
    Load CameraArray from TOML file.

    The TOML file must contain camera data indexed by port, with each camera
    entry containing intrinsics (matrix, distortions), extrinsics (rotation,
    translation), and metadata (error, grid_count, etc.).

    Rotation is stored as a 3x1 Rodrigues vector in TOML, but may be 3x3 matrix
    in legacy-migrated data. This function handles both formats.

    Args:
        path: Path to camera_array.toml

    Returns:
        CameraArray instance with all camera data loaded

    Raises:
        PersistenceError: If file doesn't exist, is invalid TOML, missing required
                         fields, or contains malformed numpy array data
    """
    if not path.exists():
        raise PersistenceError(f"CameraArray file not found: {path}")

    try:
        data = rtoml.load(path)
    except Exception as e:
        raise PersistenceError(f"Failed to load CameraArray from {path}: {e}") from e

    # Handle empty camera array file
    if not data or "cameras" not in data:
        return CameraArray({})

    cameras_dict = {}
    for port_str, camera_data in data["cameras"].items():
        logger.info(f"Loading Port {port_str}: {camera_data}")
        try:
            port = int(port_str)

            # Convert numpy arrays from lists
            matrix = _list_to_array(camera_data.get("matrix"))
            distortions = _list_to_array(camera_data.get("distortions"))
            translation = _list_to_array(camera_data.get("translation"))

            # Handle rotation: may be 3x3 matrix (legacy) or 3x1 Rodrigues (new)
            rotation_raw = _list_to_array(camera_data.get("rotation"))
            if rotation_raw is not None:
                rotation_array = rotation_raw
                if rotation_array.shape == (3, 3):
                    # Legacy format: 3x3 matrix, convert to Rodrigues
                    rotation = cv2.Rodrigues(rotation_array)[0][:, 0]
                elif rotation_array.shape in [(3,), (3, 1)]:
                    # New format: already Rodrigues vector
                    rotation = rotation_array.flatten()
                else:
                    raise ValueError(f"Invalid rotation shape: {rotation_array.shape}")
            else:
                rotation = None

            camera = CameraData(
                port=port,
                size=camera_data["size"],
                rotation_count=camera_data.get("rotation_count", 0),
                error=camera_data.get("error"),
                matrix=matrix,
                distortions=distortions,
                exposure=camera_data.get("exposure"),
                grid_count=camera_data.get("grid_count"),
                ignore=camera_data.get("ignore", False),
                translation=translation,
                rotation=rotation,
                fisheye=camera_data.get("fisheye", False),
            )
            cameras_dict[port] = camera

        except Exception as e:
            raise PersistenceError(f"Failed to parse camera {port_str}: {e}") from e

    return CameraArray(cameras_dict)


def save_camera_array(camera_array: CameraArray, path: Path) -> None:
    """
    Save CameraArray to TOML file.

    Serializes all CameraData objects to TOML format. Numpy arrays are converted
    to lists for compatibility. Rotation is stored as 3x1 Rodrigues vector.

    Args:
        camera_array: CameraArray to serialize
        path: Target file path (parent directories must exist)

    Raises:
        PersistenceError: If serialization fails or file cannot be written
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        cameras_data = {}
        for port, camera in camera_array.cameras.items():
            # Convert rotation from 3x3 matrix to 3x1 Rodrigues vector for storage
            rotation_for_config = None
            if camera.rotation is not None and camera.rotation.any():
                rotation_for_config = cv2.Rodrigues(camera.rotation)[0][:, 0].tolist()

            camera_dict = {
                "port": camera.port,
                "size": camera.size,
                "rotation_count": camera.rotation_count,
                "error": camera.error,
                "matrix": _array_to_list(camera.matrix),
                "distortions": _array_to_list(camera.distortions),
                "translation": _array_to_list(camera.translation),
                "rotation": rotation_for_config,
                "exposure": camera.exposure,
                "grid_count": camera.grid_count,
                "fisheye": camera.fisheye,
            }
            cameras_data[str(port)] = camera_dict

        data = {"cameras": cameras_data}
        _write_toml(data, path)

    except Exception as e:
        raise PersistenceError(f"Failed to save CameraArray to {path}: {e}") from e


def load_charuco(path: Path) -> Charuco:
    """
    Load Charuco board definition from TOML file.

    Args:
        path: Path to charuco.toml

    Returns:
        Charuco instance with board parameters

    Raises:
        PersistenceError: If file doesn't exist or contains invalid board parameters
    """
    if not path.exists():
        raise PersistenceError(f"Charuco file not found: {path}")

    try:
        data = rtoml.load(path)
        return Charuco(**data)
    except Exception as e:
        raise PersistenceError(f"Failed to load Charuco from {path}: {e}") from e


def save_charuco(charuco: Charuco, path: Path) -> None:
    """
    Save Charuco board definition to TOML file.

    Args:
        charuco: Charuco to serialize
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    try:
        _write_toml(charuco.__dict__, path)
    except Exception as e:
        raise PersistenceError(f"Failed to save Charuco to {path}: {e}") from e


def load_point_estimates(path: Path) -> PointEstimates:
    """
    Load PointEstimates from TOML file.

    PointEstimates contains 2D-3D correspondences for bundle adjustment:
    sync_indices, camera_indices, point_id, img (2D points), obj_indices,
    and obj (3D points).

    Args:
        path: Path to point_estimates.toml

    Returns:
        PointEstimates instance

    Raises:
        PersistenceError: If file doesn't exist or numpy arrays cannot be
                         reconstructed from stored lists
    """
    raise NotImplementedError("load_point_estimates not yet implemented")


def save_point_estimates(point_estimates: PointEstimates, path: Path) -> None:
    """
    Save PointEstimates to TOML file.

    Converts numpy arrays to lists for TOML compatibility. Write is atomic.

    Args:
        point_estimates: PointEstimates to serialize
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    raise NotImplementedError("save_point_estimates not yet implemented")


def load_stereo_pairs(path: Path) -> PairedPoseNetwork:
    """
    Load PairedPoseNetwork from TOML file.

    The file stores only directly calibrated stereo pairs (primary_port <
    secondary_port). Bridged pairs are reconstructed in-memory during load.

    Args:
        path: Path to stereo_pairs.toml

    Returns:
        PairedPoseNetwork with complete graph

    Raises:
        PersistenceError: If file doesn't exist or format is invalid
    """
    raise NotImplementedError("load_stereo_pairs not yet implemented")


def save_stereo_pairs(paired_pose_network: PairedPoseNetwork, path: Path) -> None:
    """
    Save PairedPoseNetwork to TOML file.

    Only stores raw calibrated pairs (primary_port < secondary_port) to avoid
    duplication. The full graph can be reconstructed on load.

    Args:
        paired_pose_network: PairedPoseNetwork to serialize
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    raise NotImplementedError("save_stereo_pairs not yet implemented")


# ============================================================================
# Metadata Objects
# ============================================================================


def load_capture_volume_metadata(path: Path) -> dict[str, Any]:
    """
    Load capture volume metadata from TOML file.

    Metadata includes: stage (optimization stage), origin_sync_index
    (frame index where origin was set), and other capture volume configuration.

    Args:
        path: Path to capture_volume.toml

    Returns:
        Dictionary of metadata

    Raises:
        PersistenceError: If file doesn't exist or format is invalid
    """
    raise NotImplementedError("load_capture_volume_metadata not yet implemented")


def save_capture_volume_metadata(metadata: dict[str, Any], path: Path) -> None:
    """
    Save capture volume metadata to TOML file.

    Args:
        metadata: Metadata dictionary
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    raise NotImplementedError("save_capture_volume_metadata not yet implemented")


def load_project_settings(path: Path) -> dict[str, Any]:
    """
    Load project settings from TOML file.

    Settings include: fps_sync_stream_processing, save_tracked_points_video,
    camera_count, creation_date, and other project configuration.

    Args:
        path: Path to project_settings.toml

    Returns:
        Dictionary of settings

    Raises:
        PersistenceError: If file doesn't exist or format is invalid
    """
    raise NotImplementedError("load_project_settings not yet implemented")


def save_project_settings(settings: dict[str, Any], path: Path) -> None:
    """
    Save project settings to TOML file.

    Args:
        settings: Settings dictionary
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    raise NotImplementedError("save_project_settings not yet implemented")


# ============================================================================
# CSV I/O
# ============================================================================


def load_image_points_csv(path: Path) -> pd.DataFrame:
    """
    Load 2D image points from CSV file.

    Expected columns: sync_index, port, point_id, img_loc_x, img_loc_y

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with validated image point data

    Raises:
        PersistenceError: If file doesn't exist, CSV is malformed, or data fails
                         validation against ImagePointSchema
    """
    raise NotImplementedError("load_image_points_csv not yet implemented")


def save_image_points_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save 2D image points to CSV file.

    Args:
        df: DataFrame with image point data (must match ImagePointSchema)
        path: Target CSV file path

    Raises:
        PersistenceError: If validation fails or file cannot be written
    """
    raise NotImplementedError("save_image_points_csv not yet implemented")


def load_world_points_csv(path: Path) -> pd.DataFrame:
    """
    Load 3D world points from CSV file.

    Expected columns: sync_index, point_id, x_coord, y_coord, z_coord

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with validated world point data

    Raises:
        PersistenceError: If file doesn't exist, CSV is malformed, or data fails
                         validation against WorldPointSchema
    """
    raise NotImplementedError("load_world_points_csv not yet implemented")


def save_world_points_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save 3D world points to CSV file.

    Args:
        df: DataFrame with world point data (must match WorldPointSchema)
        path: Target CSV file path

    Raises:
        PersistenceError: If validation fails or file cannot be written
    """
    raise NotImplementedError("save_world_points_csv not yet implemented")


# ============================================================================
# Legacy Migration
# ============================================================================


def migrate_legacy_config(legacy_config_path: Path, target_dir: Path) -> None:
    """
    Migrate monolithic config.toml to per-object files.

    Reads a legacy config.toml and splits it into separate files:
    - camera_array.toml
    - charuco.toml
    - point_estimates.toml (if present)
    - stereo_pairs.toml (if stereo pairs present)
    - capture_volume.toml (if capture volume metadata present)
    - project_settings.toml

    Target directory must exist. Existing files will be overwritten.

    Args:
        legacy_config_path: Path to old config.toml
        target_dir: Workspace directory for new files

    Raises:
        PersistenceError: If migration fails at any step
    """
    raise NotImplementedError("migrate_legacy_config not yet implemented")
