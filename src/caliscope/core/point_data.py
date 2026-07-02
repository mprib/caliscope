# caliscope/core/point_data.py

from __future__ import annotations
import logging
from collections.abc import Iterable
from pathlib import Path
from time import time
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from collections import defaultdict
from scipy.signal import butter, filtfilt
from caliscope.cameras.camera_array import CameraArray
from dataclasses import dataclass

logger = logging.getLogger(__name__)

STATIC_SYNC_INDEX = -1


#####################################################################################
# DLT triangulation via SVD.
# Original implementation drew from Anipose (BSD-2-Clause), created by
# Lili Karashchuk (https://github.com/lambdaloop/aniposelib/).
# Rewritten as batched numpy SVD (grouped by camera set) to drop the numba
# dependency while matching JIT performance on calibration-scale data.
#
# BSD 2-Clause License
# Copyright (c) 2019, Lili Karashchuk. All rights reserved.
# See original license text in git history or at the Anipose repository.
#####################################################################################


def triangulate_sync_index(
    projection_matrices: dict[int, np.ndarray],
    camera_ids: np.ndarray,
    object_ids: np.ndarray,
    keypoint_ids: np.ndarray,
    img_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate 2D observations to 3D points for a single sync index.

    Groups observations by (object_id, keypoint_id), then runs batched SVD
    per camera-set group. Returns (object_ids, keypoint_ids, xyz).
    """
    if len(keypoint_ids) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.zeros((0, 3))

    # Group observations by (object_id, keypoint_id) composite key
    sort_idx = np.lexsort((keypoint_ids, object_ids))
    sorted_obj = object_ids[sort_idx]
    sorted_kp = keypoint_ids[sort_idx]
    sorted_cams = camera_ids[sort_idx]
    sorted_xy = img_xy[sort_idx]

    # Find group boundaries (where either object_id or keypoint_id changes)
    breaks = np.where((np.diff(sorted_obj) != 0) | (np.diff(sorted_kp) != 0))[0] + 1
    groups_obj = np.split(sorted_obj, breaks)
    groups_kp = np.split(sorted_kp, breaks)
    groups_cams = np.split(sorted_cams, breaks)
    groups_xy = np.split(sorted_xy, breaks)

    # Filter to groups with 2+ observations (need at least 2 views to triangulate)
    valid = [(o[0], k[0], c, xy) for o, k, c, xy in zip(groups_obj, groups_kp, groups_cams, groups_xy) if len(c) > 1]

    if not valid:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.zeros((0, 3))

    # Group by exact camera set: points seen by the same cameras share
    # identical A-matrix structure (same projection matrix rows).
    # CRITICAL: sort xy to match the sorted camera order so that
    # xy[j] corresponds to cam_key[j]'s projection matrix.
    by_cam_set: dict[tuple[int, ...], list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    for oid, kid, cams, xy in valid:
        sort_order = np.argsort(cams)
        cam_key = tuple(cams[sort_order])
        by_cam_set[cam_key].append((oid, kid, xy[sort_order]))

    result_obj: list[np.ndarray] = []
    result_kp: list[np.ndarray] = []
    result_xyz: list[np.ndarray] = []

    for cam_key, entries in by_cam_set.items():
        n_cams = len(cam_key)
        n_points = len(entries)

        P_row0 = np.empty((n_cams, 4))
        P_row1 = np.empty((n_cams, 4))
        P_row2 = np.empty((n_cams, 4))
        for j, cam_id in enumerate(cam_key):
            P = projection_matrices[cam_id]
            P_row0[j] = P[0]
            P_row1[j] = P[1]
            P_row2[j] = P[2]

        batch_obj = np.array([e[0] for e in entries], dtype=np.int64)
        batch_kp = np.array([e[1] for e in entries], dtype=np.int64)
        batch_xy = np.array([e[2] for e in entries])  # (n_points, n_cams, 2)

        x = batch_xy[:, :, 0]  # (n_points, n_cams)
        y = batch_xy[:, :, 1]

        even_rows = x[:, :, None] * P_row2[None, :, :] - P_row0[None, :, :]
        odd_rows = y[:, :, None] * P_row2[None, :, :] - P_row1[None, :, :]

        A_batch = np.empty((n_points, 2 * n_cams, 4))
        A_batch[:, 0::2, :] = even_rows
        A_batch[:, 1::2, :] = odd_rows

        _, _, vh = np.linalg.svd(A_batch, full_matrices=False)
        xyzw = vh[:, -1, :]
        xyz = xyzw[:, :3] / xyzw[:, 3:4]

        result_obj.append(batch_obj)
        result_kp.append(batch_kp)
        result_xyz.append(xyz)

    return np.concatenate(result_obj), np.concatenate(result_kp), np.vstack(result_xyz)


def triangulate_image_points(
    projection_matrices: dict[int, np.ndarray],
    sync_indices: np.ndarray,
    camera_ids: np.ndarray,
    object_ids: np.ndarray,
    keypoint_ids: np.ndarray,
    img_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate all 2D observations across all sync indices in bulk.

    Groups by (sync_index, object_id, keypoint_id) composite key, batches SVD
    by camera set. Returns (sync_indices, object_ids, keypoint_ids, xyz).
    """
    n_obs = len(keypoint_ids)
    if n_obs < 2:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.zeros((0, 3)),
        )

    # Sort by (sync_index, object_id, keypoint_id) to find unique 3D points
    sort_idx = np.lexsort((keypoint_ids, object_ids, sync_indices))
    s_sync = sync_indices[sort_idx]
    s_obj = object_ids[sort_idx]
    s_kp = keypoint_ids[sort_idx]
    s_cam = camera_ids[sort_idx]
    s_xy = img_xy[sort_idx]

    # Find where any component of the composite key changes
    diff_sync = np.diff(s_sync) != 0
    diff_obj = np.diff(s_obj) != 0
    diff_kp = np.diff(s_kp) != 0
    breaks = np.where(diff_sync | diff_obj | diff_kp)[0] + 1

    grp_sync = np.split(s_sync, breaks)
    grp_obj = np.split(s_obj, breaks)
    grp_kp = np.split(s_kp, breaks)
    grp_cam = np.split(s_cam, breaks)
    grp_xy = np.split(s_xy, breaks)

    # Group by exact camera set
    by_cam_set: dict[tuple[int, ...], list[tuple[int, int, int, np.ndarray]]] = defaultdict(list)

    for gs, go, gk, gc, gxy in zip(grp_sync, grp_obj, grp_kp, grp_cam, grp_xy):
        if len(gc) < 2:
            continue
        # CRITICAL: sort xy to match the sorted camera order
        sort_order = np.argsort(gc)
        cam_key = tuple(gc[sort_order])
        by_cam_set[cam_key].append((gs[0], go[0], gk[0], gxy[sort_order]))

    if not by_cam_set:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.zeros((0, 3)),
        )

    result_sync: list[np.ndarray] = []
    result_obj: list[np.ndarray] = []
    result_kp: list[np.ndarray] = []
    result_xyz: list[np.ndarray] = []

    for cam_key, entries in by_cam_set.items():
        n_cams = len(cam_key)
        n_points = len(entries)

        P_row0 = np.empty((n_cams, 4))
        P_row1 = np.empty((n_cams, 4))
        P_row2 = np.empty((n_cams, 4))
        for j, cam_id in enumerate(cam_key):
            P = projection_matrices[cam_id]
            P_row0[j] = P[0]
            P_row1[j] = P[1]
            P_row2[j] = P[2]

        batch_sync = np.array([e[0] for e in entries], dtype=np.int64)
        batch_obj = np.array([e[1] for e in entries], dtype=np.int64)
        batch_kp = np.array([e[2] for e in entries], dtype=np.int64)
        batch_xy = np.array([e[3] for e in entries])  # (n_points, n_cams, 2)

        x = batch_xy[:, :, 0]
        y = batch_xy[:, :, 1]

        even_rows = x[:, :, None] * P_row2[None, :, :] - P_row0[None, :, :]
        odd_rows = y[:, :, None] * P_row2[None, :, :] - P_row1[None, :, :]

        A_batch = np.empty((n_points, 2 * n_cams, 4))
        A_batch[:, 0::2, :] = even_rows
        A_batch[:, 1::2, :] = odd_rows

        _, _, vh = np.linalg.svd(A_batch, full_matrices=False)
        xyzw = vh[:, -1, :]
        xyz = xyzw[:, :3] / xyzw[:, 3:4]

        result_sync.append(batch_sync)
        result_obj.append(batch_obj)
        result_kp.append(batch_kp)
        result_xyz.append(xyz)

    return (
        np.concatenate(result_sync),
        np.concatenate(result_obj),
        np.concatenate(result_kp),
        np.vstack(result_xyz),
    )


############################################################################################


def _undistort_batch(xy_df: pd.DataFrame, camera_array: CameraArray) -> pd.DataFrame:
    """Module-private helper to undistort all points in a DataFrame."""
    undistorted_points = []
    for cam_id, camera in camera_array.cameras.items():
        subset_xy = xy_df.query(f"cam_id == {cam_id}").copy()
        if not subset_xy.empty:
            points = np.vstack([subset_xy["img_loc_x"], subset_xy["img_loc_y"]]).T
            undistorted_xy = camera.undistort_points(points, output="normalized")
            subset_xy["img_loc_undistort_x"] = undistorted_xy[:, 0]
            subset_xy["img_loc_undistort_y"] = undistorted_xy[:, 1]
            undistorted_points.append(subset_xy)

    if not undistorted_points:
        return pd.DataFrame()

    xy_undistorted_df = pd.concat(undistorted_points)
    return xy_undistorted_df


# Column name constants (serve as both validation spec and column name registry).
# Call sites that only need column names use .keys().
IMAGE_POINT_COLUMNS: dict[str, dict] = {
    "sync_index": {"dtype": "int", "nullable": False},
    "cam_id": {"dtype": "int", "nullable": False},
    "object_id": {"dtype": "int", "nullable": False},
    "keypoint_id": {"dtype": "int", "nullable": False},
    "img_loc_x": {"dtype": "float", "nullable": False},
    "img_loc_y": {"dtype": "float", "nullable": False},
    "obj_loc_x": {"dtype": "float", "nullable": True},
    "obj_loc_y": {"dtype": "float", "nullable": True},
    "obj_loc_z": {"dtype": "float", "nullable": True},
}

WORLD_POINT_COLUMNS: dict[str, dict] = {
    "sync_index": {"dtype": "int", "nullable": False},
    "object_id": {"dtype": "int", "nullable": False},
    "keypoint_id": {"dtype": "int", "nullable": False},
    "x_coord": {"dtype": "float", "nullable": False},
    "y_coord": {"dtype": "float", "nullable": False},
    "z_coord": {"dtype": "float", "nullable": False},
    "frame_time": {"dtype": "float", "nullable": True},
}


def _validate_dataframe(
    df: pd.DataFrame,
    schema: dict[str, dict],
    schema_name: str,
) -> pd.DataFrame:
    """Validate and coerce a DataFrame against a column schema.

    Checks that all required columns are present, coerces types, and verifies
    non-nullable columns contain no NaN/None values. Extra columns are allowed
    (strict=False equivalent).

    Returns the coerced DataFrame. Raises ValueError on validation failure.
    """
    # 1. Check column presence
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise ValueError(
            f"{schema_name} validation failed: column(s) {missing} not in dataframe. Columns found: {list(df.columns)}"
        )

    # 2. Coerce types
    for col, spec in schema.items():
        if spec["dtype"] == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif spec["dtype"] == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Check nullability
    for col, spec in schema.items():
        if not spec["nullable"] and df[col].isna().any():
            n_null = df[col].isna().sum()
            raise ValueError(
                f"{schema_name} validation failed: non-nullable column '{col}' contains {n_null} null value(s)"
            )

    # 4. Downcast non-nullable Int64 → int64 for numpy compatibility.
    # Int64 (nullable extension type) produces object arrays from .to_numpy().
    # After confirming no nulls, downcast to standard int64 for clean numpy interop.
    for col, spec in schema.items():
        if spec["dtype"] == "int" and not spec["nullable"]:
            df[col] = df[col].astype("int64")

    return df


class ImagePoints:
    """A validated, immutable container for 2D (x,y) point data."""

    _df: pd.DataFrame

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    def __init__(self, df: pd.DataFrame):
        # Ensure optional columns exist even if not in source data
        df = df.copy()  # Don't modify the original DataFrame
        for col in ["obj_loc_x", "obj_loc_y", "obj_loc_z", "frame_time"]:
            if col not in df.columns:
                df[col] = np.nan

        self._df = _validate_dataframe(df, IMAGE_POINT_COLUMNS, "ImagePoints")

        # Warn about duplicate keys that could cause incorrect triangulation
        key_cols = ["sync_index", "cam_id", "object_id", "keypoint_id"]
        dupes = self._df.duplicated(subset=key_cols)
        if dupes.any():
            n = int(dupes.sum())
            logger.warning(
                f"ImagePoints contains {n} duplicate ({', '.join(key_cols)}) rows. "
                f"Duplicates may cause incorrect triangulation results."
            )

    @classmethod
    def from_csv(cls, path: str | Path) -> ImagePoints:
        df = pd.read_csv(path)
        # Constructor handles adding missing optional columns
        return cls(df)

    def to_csv(self, path: str | Path) -> None:
        """Save image points to CSV file.

        Uses atomic write (temp file + fsync + rename) to prevent data loss.

        Raises:
            PersistenceError: If write fails
        """
        from caliscope.persistence import _safe_write_csv, CSV_FLOAT_PRECISION, PersistenceError

        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _safe_write_csv(self._df, path, index=False, float_format=CSV_FLOAT_PRECISION)
        except Exception as e:
            raise PersistenceError(f"Failed to save image points to {path}: {e}") from e

    def fill_gaps(self, max_gap_size: int = 3) -> ImagePoints:
        xy_filled = pd.DataFrame()
        index_key = "sync_index"
        last_cam_id = -1
        base_df = self.df
        for (cam_id, object_id, keypoint_id), group in base_df.groupby(["cam_id", "object_id", "keypoint_id"]):
            if last_cam_id != cam_id:
                logger.info(
                    f"Gap filling for (x,y) data from cam_id {cam_id}. "
                    f"Filling gaps that are {max_gap_size} frames or less..."
                )
            last_cam_id = cam_id
            group = group.sort_values(index_key)
            all_frames = pd.DataFrame({index_key: np.arange(group[index_key].min(), group[index_key].max() + 1)})
            all_frames["cam_id"] = int(cam_id)  # type: ignore[arg-type]
            all_frames["object_id"] = int(object_id)  # type: ignore[arg-type]
            all_frames["keypoint_id"] = int(keypoint_id)  # type: ignore[arg-type]
            merged = pd.merge(all_frames, group, on=["cam_id", "object_id", "keypoint_id", index_key], how="left")
            merged["gap_size"] = (
                merged["img_loc_x"].isnull().astype(int).groupby((merged["img_loc_x"].notnull()).cumsum()).cumsum()
            )
            merged = merged[merged["gap_size"] <= max_gap_size]
            for col in ["img_loc_x", "img_loc_y", "frame_time"]:
                if col in merged.columns:
                    merged[col] = merged[col].interpolate(method="linear", limit=max_gap_size)
            xy_filled = pd.concat([xy_filled, merged])
        logger.info("(x,y) gap filling complete")
        return ImagePoints(xy_filled.dropna(subset=["img_loc_x"]))

    def filter_to_objects(self, object_ids: Iterable[int]) -> ImagePoints:
        """Return a copy containing only rows whose object_id is in object_ids."""
        valid = set(object_ids)
        mask = self._df["object_id"].isin(valid)
        dropped = self._df[~mask]
        if len(dropped) > 0:
            dropped_ids = sorted(dropped["object_id"].unique())
            for oid in dropped_ids:
                n = int((dropped["object_id"] == oid).sum())
                logger.info(f"filter_to_objects: dropped object_id={oid} ({n} rows)")
        return ImagePoints(self._df[mask].copy())

    def triangulate(
        self,
        camera_array: CameraArray,
        static_object_ids: frozenset[int] = frozenset(),
    ) -> WorldPoints:
        """Triangulates 2D points to create 3D points using the provided CameraArray.

        Observations whose object_id is in static_object_ids are treated as a single
        rigid object across all frames: their sync_index is ignored and all observations
        of a given (object_id, keypoint_id) are triangulated together into one 3D point,
        stored under STATIC_SYNC_INDEX.
        """
        xy_df = self.df
        if xy_df.empty:
            return WorldPoints(pd.DataFrame(columns=list(WORLD_POINT_COLUMNS.keys())))

        # Only process cameras that are both in data AND posed
        cam_ids_in_data = xy_df["cam_id"].unique()
        posed_cam_ids = list(camera_array.posed_cam_id_to_index.keys())
        valid_cam_ids = [c for c in cam_ids_in_data if c in posed_cam_ids]

        if not valid_cam_ids:
            logger.warning("No cameras in data have extrinsics for triangulation")
            return WorldPoints(pd.DataFrame(columns=list(WORLD_POINT_COLUMNS.keys())))

        normalized_projection_matrices = camera_array.normalized_projection_matrices

        # Undistort all image points before triangulation
        undistorted_xy = _undistort_batch(xy_df, camera_array)

        # Filter to valid cameras
        mask = undistorted_xy["cam_id"].isin(valid_cam_ids)
        valid_data = undistorted_xy[mask]

        if valid_data.empty:
            return WorldPoints(pd.DataFrame(columns=list(WORLD_POINT_COLUMNS.keys())))

        # Compute mean frame_time per sync_index
        frame_times = xy_df.groupby("sync_index")["frame_time"].mean()

        logger.info("Beginning bulk triangulation across all sync indices...")
        start = time()

        result_parts: list[pd.DataFrame] = []

        if static_object_ids:
            static_mask = valid_data["object_id"].isin(static_object_ids)
            mobile_data = valid_data[~static_mask]
            static_data = valid_data[static_mask]
        else:
            mobile_data = valid_data
            static_data = valid_data.iloc[0:0]

        if not mobile_data.empty:
            sync_arr = mobile_data["sync_index"].to_numpy()
            cam_arr = mobile_data["cam_id"].to_numpy()
            obj_arr = mobile_data["object_id"].to_numpy()
            kp_arr = mobile_data["keypoint_id"].to_numpy()
            xy_arr = np.column_stack(
                [
                    mobile_data["img_loc_undistort_x"].to_numpy(),
                    mobile_data["img_loc_undistort_y"].to_numpy(),
                ]
            )

            out_sync, out_obj, out_kp, out_xyz = triangulate_image_points(
                normalized_projection_matrices,
                sync_arr,
                cam_arr,
                obj_arr,
                kp_arr,
                xy_arr,
            )

            if len(out_kp) > 0:
                out_frame_times = frame_times.reindex(out_sync).to_numpy()
                result_parts.append(
                    pd.DataFrame(
                        {
                            "sync_index": out_sync,
                            "object_id": out_obj,
                            "keypoint_id": out_kp,
                            "x_coord": out_xyz[:, 0],
                            "y_coord": out_xyz[:, 1],
                            "z_coord": out_xyz[:, 2],
                            "frame_time": out_frame_times,
                        }
                    )
                )

        if not static_data.empty:
            # Remap sync_index to the sentinel so all observations of each
            # (object_id, keypoint_id) group into a single triangulation.
            sync_arr = np.full(len(static_data), STATIC_SYNC_INDEX, dtype=np.int64)
            cam_arr = static_data["cam_id"].to_numpy()
            obj_arr = static_data["object_id"].to_numpy()
            kp_arr = static_data["keypoint_id"].to_numpy()
            xy_arr = np.column_stack(
                [
                    static_data["img_loc_undistort_x"].to_numpy(),
                    static_data["img_loc_undistort_y"].to_numpy(),
                ]
            )

            out_sync, out_obj, out_kp, out_xyz = triangulate_image_points(
                normalized_projection_matrices,
                sync_arr,
                cam_arr,
                obj_arr,
                kp_arr,
                xy_arr,
            )

            if len(out_kp) > 0:
                logger.info(f"Triangulated {len(out_kp)} static world points from {len(static_data)} observations")
                result_parts.append(
                    pd.DataFrame(
                        {
                            "sync_index": out_sync,
                            "object_id": out_obj,
                            "keypoint_id": out_kp,
                            "x_coord": out_xyz[:, 0],
                            "y_coord": out_xyz[:, 1],
                            "z_coord": out_xyz[:, 2],
                            "frame_time": np.full(len(out_kp), np.nan),
                        }
                    )
                )

        elapsed = time() - start

        if not result_parts:
            logger.info(f"Triangulation complete: 0 points in {elapsed:.2f}s")
            return WorldPoints(pd.DataFrame(columns=list(WORLD_POINT_COLUMNS.keys())))

        xyz_df = pd.concat(result_parts, ignore_index=True)
        n_static = int((xyz_df["sync_index"] == STATIC_SYNC_INDEX).sum())
        n_mobile_sync = xyz_df.loc[xyz_df["sync_index"] != STATIC_SYNC_INDEX, "sync_index"].nunique()
        logger.info(
            f"Triangulation complete: {len(xyz_df)} points from "
            f"{n_mobile_sync} sync indices (+{n_static} static) in {elapsed:.2f}s"
        )

        return WorldPoints(xyz_df)


@dataclass(frozen=True)
class WorldPoints:
    """A validated, immutable container for 3D (x,y,z) point data."""

    _df: pd.DataFrame
    min_index: int | None = None
    max_index: int | None = None

    def __post_init__(self):
        # Validate schema
        validated = _validate_dataframe(self._df.copy(), WORLD_POINT_COLUMNS, "WorldPoints")
        object.__setattr__(self, "_df", validated)

        # Warn about duplicate keys that could cause incorrect results
        key_cols = ["sync_index", "object_id", "keypoint_id"]
        dupes = self._df.duplicated(subset=key_cols)
        if dupes.any():
            n = int(dupes.sum())
            logger.warning(
                f"WorldPoints contains {n} duplicate ({', '.join(key_cols)}) rows. "
                f"Duplicates may cause incorrect results."
            )

        # calculate start and stop index, excluding static points (sentinel sync_index)
        non_static = self._df["sync_index"] != STATIC_SYNC_INDEX
        if non_static.any():
            min_index = int(self._df.loc[non_static, "sync_index"].min())
            max_index = int(self._df.loc[non_static, "sync_index"].max())
        else:
            min_index = 0
            max_index = 0
        object.__setattr__(self, "min_index", min_index)
        object.__setattr__(self, "max_index", max_index)

    @property
    def df(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame to maintain immutability."""
        return self._df.copy()

    @property
    def points(self) -> NDArray:
        """Return Nx3 numpy array of coordinates."""
        return self._df[["x_coord", "y_coord", "z_coord"]].values

    def fill_gaps(self, max_gap_size: int = 3) -> WorldPoints:
        """Fill gaps in 3D point trajectories."""
        xyz_filled = pd.DataFrame()
        base_df = self.df

        for (object_id, keypoint_id), group in base_df.groupby(["object_id", "keypoint_id"]):
            group = group.sort_values("sync_index")
            all_frames = pd.DataFrame(
                {"sync_index": np.arange(group["sync_index"].min(), group["sync_index"].max() + 1)}
            )
            all_frames["object_id"] = int(object_id)  # type: ignore[arg-type]
            all_frames["keypoint_id"] = int(keypoint_id)  # type: ignore[arg-type]
            merged = pd.merge(all_frames, group, on=["object_id", "keypoint_id", "sync_index"], how="left")

            # Calculate gap size
            merged["gap_size"] = (
                merged["x_coord"].isnull().astype(int).groupby((merged["x_coord"].notnull()).cumsum()).cumsum()
            )
            merged = merged[merged["gap_size"] <= max_gap_size]

            # Interpolate coordinates and frame_time
            for col in ["x_coord", "y_coord", "z_coord", "frame_time"]:
                if col in merged.columns:
                    merged[col] = merged[col].interpolate(method="linear", limit=max_gap_size)

            xyz_filled = pd.concat([xyz_filled, merged])

        # Return new WorldPoints instance (immutable pattern)
        return WorldPoints(xyz_filled.dropna(subset=["x_coord"]))

    def smooth(self, fps: float, cutoff_freq: float, order: int = 2) -> WorldPoints:
        """Apply Butterworth filter to smooth 3D trajectories."""
        # output="ba" returns (b, a) coefficients; scipy stubs don't narrow this
        b, a = butter(order, cutoff_freq, btype="low", fs=fps, output="ba")  # type: ignore[assignment]
        base_df = self.df
        xyz_filtered = base_df.copy()

        for (object_id, keypoint_id), group in base_df.groupby(["object_id", "keypoint_id"]):
            if group.shape[0] > 3 * order:
                xyz_filtered.loc[group.index, "x_coord"] = filtfilt(b, a, group["x_coord"])
                xyz_filtered.loc[group.index, "y_coord"] = filtfilt(b, a, group["y_coord"])
                xyz_filtered.loc[group.index, "z_coord"] = filtfilt(b, a, group["z_coord"])

        # Return new WorldPoints instance (immutable pattern)
        return WorldPoints(xyz_filtered)

    @classmethod
    def from_csv(cls, path: str | Path) -> WorldPoints:
        """Load WorldPoints from a CSV file.

        Expected columns: sync_index, object_id, keypoint_id, x_coord, y_coord, z_coord
        Optional: frame_time (nullable)
        """
        df = pd.read_csv(path)
        return cls(df)  # Constructor handles validation on construction

    def to_csv(self, path: str | Path) -> None:
        """Save world points to CSV file.

        Uses atomic write (temp file + fsync + rename) to prevent data loss.

        Raises:
            PersistenceError: If write fails
        """
        from caliscope.persistence import _safe_write_csv, CSV_FLOAT_PRECISION, PersistenceError

        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _safe_write_csv(self._df, path, index=False, float_format=CSV_FLOAT_PRECISION)
        except Exception as e:
            raise PersistenceError(f"Failed to save world points to {path}: {e}") from e
