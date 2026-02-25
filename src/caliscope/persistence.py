import logging
from pathlib import Path
from typing import Any
import cv2

import numpy as np
import pandas as pd
import rtoml

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.point_data import ImagePointSchema, ImagePoints, WorldPoints, WorldPointSchema
from caliscope.core.charuco import Charuco
from caliscope.core.chessboard import Chessboard
from caliscope.core.aruco_target import ArucoTarget
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.core.bootstrap_pose.stereopairs import StereoPair

logger = logging.getLogger(__name__)


class PersistenceError(Exception):
    """Raised when file I/O or data validation fails."""

    pass


CSV_FLOAT_PRECISION = "%.6f"  # 6 decimal places = micron precision at meter scale


def _clean_scalar(value: Any) -> Any:
    """
    Helper to handle TOML 'null' string artifacts or actual None values.
    Returns None if value is None or the string 'null'.
    """
    if value is None or value == "null":
        return None
    return value


def _array_to_list(arr: np.ndarray | None) -> list | None:
    """
    Convert numpy array to list for TOML serialization.

    Args:
        arr: Numpy array or None

    Returns:
        List representation or None
    """
    return arr.tolist() if arr is not None else None


def _list_to_array(lst: Any, dtype: type[np.generic] = np.float64) -> np.ndarray | None:
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


def load_camera_array(path: Path) -> CameraArray:
    """
    Load CameraArray from TOML file.

    The TOML file must contain camera data indexed by cam_id, with each camera
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
    for cam_id_str, camera_data in data["cameras"].items():
        try:
            cam_id = int(cam_id_str)

            # Convert numpy arrays from lists
            matrix = _list_to_array(camera_data.get("matrix"))
            distortions = _list_to_array(camera_data.get("distortions"))
            translation = _list_to_array(camera_data.get("translation"))

            # Handle rotation: may be 3x3 matrix (legacy) or 3x1 Rodrigues (new)
            # (Includes the fix from the previous step)
            rotation_raw = _list_to_array(camera_data.get("rotation"))
            if rotation_raw is not None:
                rotation_array = rotation_raw
                if rotation_array.shape == (3, 3):
                    rotation = rotation_array
                elif rotation_array.shape in [(3,), (3, 1)]:
                    rotation = cv2.Rodrigues(rotation_array)[0]
                else:
                    raise ValueError(f"Invalid rotation shape: {rotation_array.shape}")
            else:
                rotation = None

            camera = CameraData(
                cam_id=cam_id,
                size=(camera_data["size"][0], camera_data["size"][1]),
                rotation_count=camera_data.get("rotation_count", 0),
                # WRAP SCALARS IN _clean_scalar
                error=_clean_scalar(camera_data.get("error")),
                matrix=matrix,
                distortions=distortions,
                exposure=_clean_scalar(camera_data.get("exposure")),
                grid_count=_clean_scalar(camera_data.get("grid_count")),
                ignore=camera_data.get("ignore", False),
                translation=translation,
                rotation=rotation,
                fisheye=camera_data.get("fisheye", False),
            )
            cameras_dict[cam_id] = camera

        except Exception as e:
            raise PersistenceError(f"Failed to parse camera {cam_id_str}: {e}") from e

    return CameraArray(cameras_dict)


def save_camera_array(camera_array: CameraArray, path: Path) -> None:
    """
    Save CameraArray to TOML file.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        cameras_data = {}
        for cam_id, camera in camera_array.cameras.items():
            # Convert rotation from 3x3 matrix to 3x1 Rodrigues vector for storage
            rotation_for_config = None
            if camera.rotation is not None and camera.rotation.any():
                rotation_for_config = cv2.Rodrigues(camera.rotation)[0][:, 0].tolist()

            camera_dict = {
                "cam_id": camera.cam_id,
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

            # In TOML, a missing key is the correct way to represent "None/Null".
            # This prevents rtoml from writing "null" strings.
            clean_camera_dict = {k: v for k, v in camera_dict.items() if v is not None}

            cameras_data[str(cam_id)] = clean_camera_dict

        data = {"cameras": cameras_data}
        _write_toml(data, path)

    except Exception as e:
        raise PersistenceError(f"Failed to save CameraArray to {path}: {e}") from e


def save_camera_array_aniposelib(camera_array: CameraArray, path: Path) -> None:
    """
    Save CameraArray in aniposelib-compatible TOML format.

    Only exports posed cameras (those with both rotation and translation).
    Uses top-level [cam_N] sections instead of nested structure.

    Example output format:
        [cam_0]
        name = "cam_0"
        size = [1280, 720]
        matrix = [[903.5, 0.0, 618.3], [0.0, 907.8, 394.2], [0.0, 0.0, 1.0]]
        distortions = [-0.332, 0.046, -0.004, 0.004, 0.066]
        rotation = [1.234, -0.567, 0.890]  # Rodrigues vector (3 elements)
        translation = [0.171, -0.032, 1.208]

        [metadata]
        adjusted = false
        error = 0.0

    Args:
        camera_array: CameraArray to export
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {}

        # Export only posed cameras
        for cam_id, camera in camera_array.posed_cameras.items():
            # Convert 3x3 rotation matrix to 3x1 Rodrigues vector
            rotation_rodrigues = None
            if camera.rotation is not None and camera.rotation.any():
                rotation_rodrigues = cv2.Rodrigues(camera.rotation)[0][:, 0].tolist()

            # Aniposelib expects flat 1D lists for distortions and translation.
            # Flatten defensively in case upstream code delivers 2D arrays
            # (e.g. cv2.calibrateCamera returns distortions as (1, 5)).
            distortions_flat = camera.distortions.ravel().tolist() if camera.distortions is not None else None
            translation_flat = camera.translation.ravel().tolist() if camera.translation is not None else None

            camera_dict = {
                "name": f"cam_{cam_id}",
                "size": [int(camera.size[0]), int(camera.size[1])],
                "matrix": _array_to_list(camera.matrix),
                "distortions": distortions_flat,
                "rotation": rotation_rodrigues,
                "translation": translation_flat,
            }

            data[f"cam_{cam_id}"] = camera_dict

        # Add metadata section
        data["metadata"] = {"adjusted": False, "error": 0.0}

        _write_toml(data, path)
        logger.info(f"Saved aniposelib-compatible camera array to {path}")

    except Exception as e:
        raise PersistenceError(f"Failed to save aniposelib CameraArray to {path}: {e}") from e


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


def load_chessboard(path: Path) -> Chessboard:
    """
    Load Chessboard pattern definition from TOML file.

    Args:
        path: Path to chessboard.toml

    Returns:
        Chessboard instance with pattern parameters

    Raises:
        PersistenceError: If file doesn't exist or contains invalid parameters
    """
    if not path.exists():
        raise PersistenceError(f"Chessboard file not found: {path}")

    try:
        data = rtoml.load(path)
        data.pop("square_size_cm", None)  # Strip legacy field
        return Chessboard(**data)
    except Exception as e:
        raise PersistenceError(f"Failed to load Chessboard from {path}: {e}") from e


def save_chessboard(chessboard: Chessboard, path: Path) -> None:
    """
    Save Chessboard pattern definition to TOML file.

    Args:
        chessboard: Chessboard to serialize
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    from dataclasses import asdict

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_toml(asdict(chessboard), path)
    except Exception as e:
        raise PersistenceError(f"Failed to save Chessboard to {path}: {e}") from e


def load_stereo_pairs(path: Path) -> PairedPoseNetwork:
    """
    Load PairedPoseNetwork from TOML file.

    The file stores only directly calibrated stereo pairs (primary_cam_id <
    secondary_cam_id) with Rodrigues rotation vectors. On load, we:
    1. Convert Rodrigues vectors back to 3x3 rotation matrices
    2. Reconstruct the full graph by adding inverted pairs
    3. Build bridged connections for missing pairs

    Args:
        path: Path to stereo_pairs.toml

    Returns:
        PairedPoseNetwork with complete graph

    Raises:
        PersistenceError: If file doesn't exist or format is invalid
    """
    if not path.exists():
        raise PersistenceError(f"Stereo pairs file not found: {path}")

    try:
        data = rtoml.load(path)
        if not data:
            # Empty file - return empty network
            return PairedPoseNetwork({})

        raw_pairs = {}
        for key, pair_data in data.items():
            # Parse key format: "stereo_1_2"
            try:
                _, cam_id_a_str, cam_id_b_str = key.split("_")
                cam_id_a, cam_id_b = int(cam_id_a_str), int(cam_id_b_str)
            except (ValueError, AttributeError):
                logger.warning(f"Skipping invalid stereo pair key: {key}")
                continue

            # Convert Rodrigues vector to 3x3 rotation matrix
            rotation_rodrigues = _list_to_array(pair_data.get("rotation"))
            if rotation_rodrigues is None:
                logger.warning(f"Missing rotation for pair {key}, skipping")
                continue

            if rotation_rodrigues.shape != (3,):
                logger.warning(f"Invalid rotation shape for pair {key}: {rotation_rodrigues.shape}, expected (3,)")
                continue

            rotation_matrix = cv2.Rodrigues(rotation_rodrigues)[0]

            translation = _list_to_array(pair_data.get("translation"))
            if translation is None:
                logger.warning(f"Missing translation for pair {key}, skipping")
                continue

            if translation.shape != (3, 1) and translation.shape != (3,):
                logger.warning(f"Invalid translation shape for pair {key}: {translation.shape}, expected (3,1) or (3,)")
                continue

            # Ensure translation is column vector
            if translation.shape == (3,):
                translation = translation.reshape(3, 1)

            pair = StereoPair(
                primary_cam_id=cam_id_a,
                secondary_cam_id=cam_id_b,
                error_score=float(pair_data.get("RMSE", 0.0)),
                rotation=rotation_matrix,
                translation=translation,
            )
            raw_pairs[pair.pair] = pair

        # Use PairedPoseNetwork's built-in graph reconstruction
        return PairedPoseNetwork.from_raw_estimates(raw_pairs)

    except Exception as e:
        raise PersistenceError(f"Failed to load stereo pairs from {path}: {e}") from e


def save_stereo_pairs(paired_pose_network: PairedPoseNetwork, path: Path) -> None:
    """
    Save PairedPoseNetwork to TOML file.

    Only stores raw calibrated pairs (primary_cam_id < secondary_cam_id) to avoid
    duplication. Converts 3x3 rotation matrices to 3x1 Rodrigues vectors for
    storage efficiency.

    Args:
        paired_pose_network: PairedPoseNetwork to serialize
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    try:
        # Get only forward pairs (a < b) to avoid duplication
        stereo_data = {}
        for (a, b), pair in paired_pose_network._pairs.items():
            if a >= b:  # Skip inverted pairs
                continue

            # Convert 3x3 rotation matrix to 3x1 Rodrigues vector
            rotation_rodrigues = None
            if pair.rotation is not None:
                rodrigues, _ = cv2.Rodrigues(pair.rotation)
                rotation_rodrigues = rodrigues.flatten().tolist()

            # Ensure translation is list format
            translation_list = None
            if pair.translation is not None:
                # Flatten to list, handling both (3,) and (3,1) shapes
                translation_list = pair.translation.flatten().tolist()

            pair_dict = {
                "RMSE": pair.error_score,
                "rotation": rotation_rodrigues,
                "translation": translation_list,
            }

            # Filter out None values
            pair_dict = {k: v for k, v in pair_dict.items() if v is not None}

            stereo_data[f"stereo_{a}_{b}"] = pair_dict

        _write_toml(stereo_data, path)
    except Exception as e:
        raise PersistenceError(f"Failed to save stereo pairs to {path}: {e}") from e


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
    """
    Save capture volume metadata to TOML file.

    Args:
        metadata: Metadata dictionary with keys: stage, origin_sync_index
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    try:
        # Filter out None values - TOML doesn't have null type
        data_to_save = {k: v for k, v in metadata.items() if v is not None}
        _write_toml(data_to_save, path)
    except Exception as e:
        raise PersistenceError(f"Failed to save capture volume metadata to {path}: {e}") from e


def load_project_settings(path: Path) -> dict[str, Any]:
    """
    Load project settings from TOML file.

    Settings include: fps_sync_stream_processing, save_tracked_points_video,
    camera_count, creation_date, and other project configuration.

    Args:
        path: Path to project_settings.toml

    Returns:
        Dictionary of settings. Returns empty dict if file doesn't exist
        (for backward compatibility with new projects).

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
    """
    Save project settings to TOML file.

    Args:
        settings: Settings dictionary
        path: Target file path

    Raises:
        PersistenceError: If serialization or write fails
    """
    try:
        # Filter out None values
        data_to_save = {k: v for k, v in settings.items() if v is not None}
        _write_toml(data_to_save, path)
    except Exception as e:
        raise PersistenceError(f"Failed to save project settings to {path}: {e}") from e


def load_image_points_csv(path: Path) -> ImagePoints:
    """
    Load 2D image points from CSV file.

    Expected columns: sync_index, cam_id, point_id, img_loc_x, img_loc_y

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with validated image point data

    Raises:
        PersistenceError: If file doesn't exist, CSV is malformed, or data fails
                         validation against ImagePointSchema
    """
    if not path.exists():
        raise PersistenceError(f"Image points CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
        # ImagePoints constructor handles adding missing optional columns
        return ImagePoints(df)
    except Exception as e:
        raise PersistenceError(f"Failed to load image points from {path}: {e}") from e


def save_image_points_csv(image_points: ImagePoints, path: Path) -> None:
    """
    Save 2D image points to CSV file.

    Args:
        df: DataFrame with image point data (must match ImagePointSchema)
        path: Target CSV file path

    Raises:
        PersistenceError: If validation fails or file cannot be written
    """
    try:
        # Validate before saving
        validated_df: pd.DataFrame = ImagePointSchema.validate(image_points.df)
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        validated_df.to_csv(path, index=False, float_format=CSV_FLOAT_PRECISION)
    except Exception as e:
        raise PersistenceError(f"Failed to save image points to {path}: {e}") from e


def load_world_points_csv(path: Path) -> WorldPoints:
    """
    Load 3D world points from CSV file.

    Expected columns: sync_index, point_id, x_coord, y_coord, z_coord

    Args:
        path: Path to CSV file

    Returns:
        WorldPoints instance with validated data

    Raises:
        PersistenceError: If file doesn't exist, CSV is malformed, or data fails
                         validation against WorldPointSchema
    """
    if not path.exists():
        raise PersistenceError(f"World points CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
        # Validate with Pandera schema to ensure data integrity
        validated_df = WorldPointSchema.validate(df)
        return WorldPoints(validated_df)
    except Exception as e:
        raise PersistenceError(f"Failed to load world points from {path}: {e}") from e


def save_world_points_csv(world_points: WorldPoints, path: Path) -> None:
    """
    Save 3D world points to CSV file.

    Args:
        world_points: WorldPoints instance to save
        path: Target CSV file path

    Raises:
        PersistenceError: If validation fails or file cannot be written
    """
    try:
        # Validate before saving to ensure data consistency
        validated_df: pd.DataFrame = WorldPointSchema.validate(world_points.df)
        # Ensure parent directory exists (atomic file operations)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use consistent float precision for reproducibility
        validated_df.to_csv(path, index=False, float_format=CSV_FLOAT_PRECISION)
    except Exception as e:
        raise PersistenceError(f"Failed to save world points to {path}: {e}") from e


def load_aruco_target(path: Path) -> ArucoTarget:
    """Load ArucoTarget from TOML file.

    TOML format:
        dictionary = 0
        marker_size_m = 0.05

        [corners.0]
        positions = [[-0.025, -0.025, 0.0], [0.025, -0.025, 0.0], ...]
    """
    if not path.exists():
        raise PersistenceError(f"ArucoTarget file not found: {path}")

    try:
        data = rtoml.load(path)

        dictionary = data["dictionary"]
        marker_size_m = data["marker_size_m"]

        corners: dict[int, np.ndarray] = {}
        for marker_id_str, corner_data in data.get("corners", {}).items():
            marker_id = int(marker_id_str)
            positions = np.array(corner_data["positions"], dtype=np.float64)
            if positions.shape != (4, 3):
                raise ValueError(f"Marker {marker_id} has invalid shape: {positions.shape}")
            corners[marker_id] = positions

        return ArucoTarget(
            dictionary=dictionary,
            corners=corners,
            marker_size_m=marker_size_m,
        )
    except PersistenceError:
        raise
    except Exception as e:
        raise PersistenceError(f"Failed to load ArucoTarget from {path}: {e}") from e


def save_aruco_target(target: ArucoTarget, path: Path) -> None:
    """Save ArucoTarget to TOML file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        corners_data = {}
        for marker_id, positions in target.corners.items():
            corners_data[str(marker_id)] = {"positions": positions.tolist()}

        data = {
            "dictionary": target.dictionary,
            "marker_size_m": target.marker_size_m,
            "corners": corners_data,
        }

        _write_toml(data, path)
    except PersistenceError:
        raise
    except Exception as e:
        raise PersistenceError(f"Failed to save ArucoTarget to {path}: {e}") from e
