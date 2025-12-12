import logging
from pathlib import Path
from typing import Any
import cv2

import numpy as np
import pandas as pd
import rtoml

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.post_processing.point_data import ImagePointSchema, ImagePoints, WorldPoints, WorldPointSchema
from caliscope.calibration.charuco import Charuco
from caliscope.calibration.capture_volume.point_estimates import PointEstimates
from caliscope.calibration.array_initialization.paired_pose_network import PairedPoseNetwork
from caliscope.calibration.array_initialization.stereopairs import StereoPair

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
                port=port,
                size=camera_data["size"],
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
            cameras_dict[port] = camera

        except Exception as e:
            raise PersistenceError(f"Failed to parse camera {port_str}: {e}") from e

    return CameraArray(cameras_dict)


def save_camera_array(camera_array: CameraArray, path: Path) -> None:
    """
    Save CameraArray to TOML file.
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

            # In TOML, a missing key is the correct way to represent "None/Null".
            # This prevents rtoml from writing "null" strings.
            clean_camera_dict = {k: v for k, v in camera_dict.items() if v is not None}

            cameras_data[str(port)] = clean_camera_dict

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
    if not path.exists():
        raise PersistenceError(f"Point estimates file not found: {path}")

    try:
        data = rtoml.load(path)
        if not data:
            raise PersistenceError(f"Point estimates file is empty: {path}")

        # Convert all list fields back to numpy arrays
        # Use explicit dtype matching to avoid type issues
        sync_indices = _list_to_array(data.get("sync_indices"), dtype=np.int64)
        camera_indices = _list_to_array(data.get("camera_indices"), dtype=np.int64)
        point_id = _list_to_array(data.get("point_id"), dtype=np.int64)
        img = _list_to_array(data.get("img"), dtype=np.float32)
        obj_indices = _list_to_array(data.get("obj_indices"), dtype=np.int64)
        obj = _list_to_array(data.get("obj"), dtype=np.float32)

        # Validate that we have all required fields
        required_fields = ["sync_indices", "camera_indices", "point_id", "img", "obj_indices", "obj"]
        for field in required_fields:
            if locals()[field] is None:
                raise PersistenceError(f"Missing required field '{field}' in point estimates")

        # Validate array shapes and consistency
        if img.ndim != 2 or img.shape[1] != 2:
            raise PersistenceError(f"Invalid img shape: {img.shape}, expected (N, 2)")

        if obj.ndim != 2 or obj.shape[1] != 3:
            raise PersistenceError(f"Invalid obj shape: {obj.shape}, expected (N, 3)")

        n_observations = len(sync_indices)
        if not all(len(arr) == n_observations for arr in [camera_indices, point_id, obj_indices]):
            raise PersistenceError("Inconsistent array lengths in point estimates")

        if len(img) != n_observations:
            raise PersistenceError(f"img array length {len(img)} doesn't match sync_indices length {n_observations}")

        return PointEstimates(
            sync_indices=sync_indices,
            camera_indices=camera_indices,
            point_id=point_id,
            img=img,
            obj_indices=obj_indices,
            obj=obj,
        )

    except Exception as e:
        raise PersistenceError(f"Failed to load point estimates from {path}: {e}") from e


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
    try:
        # Convert all numpy arrays to lists
        data = {
            "sync_indices": point_estimates.sync_indices.tolist(),
            "camera_indices": point_estimates.camera_indices.tolist(),
            "point_id": point_estimates.point_id.tolist(),
            "img": point_estimates.img.tolist(),
            "obj_indices": point_estimates.obj_indices.tolist(),
            "obj": point_estimates.obj.tolist(),
        }

        _write_toml(data, path)
    except Exception as e:
        raise PersistenceError(f"Failed to save point estimates to {path}: {e}") from e


def load_stereo_pairs(path: Path) -> PairedPoseNetwork:
    """
    Load PairedPoseNetwork from TOML file.

    The file stores only directly calibrated stereo pairs (primary_port <
    secondary_port) with Rodrigues rotation vectors. On load, we:
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
                _, port_a_str, port_b_str = key.split("_")
                port_a, port_b = int(port_a_str), int(port_b_str)
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
                primary_port=port_a,
                secondary_port=port_b,
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

    Only stores raw calibrated pairs (primary_port < secondary_port) to avoid
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

    Expected columns: sync_index, port, point_id, img_loc_x, img_loc_y

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
        # Validate with Pandera schema
        validated_df = ImagePointSchema.validate(df)
        return ImagePoints(validated_df)
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
