"""Validated detection boxes retained between tracking and pose stages."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from caliscope.core.point_data import _validate_dataframe
from caliscope.persistence import CSV_FLOAT_PRECISION, PersistenceError, _safe_write_csv

logger = logging.getLogger(__name__)

BOX_TRACK_COLUMNS: dict[str, dict[str, object]] = {
    "sync_index": {"dtype": "int", "nullable": False},
    "cam_id": {"dtype": "int", "nullable": False},
    "frame_index": {"dtype": "int", "nullable": False},
    "frame_time": {"dtype": "float", "nullable": False},
    "track_id": {"dtype": "int", "nullable": False},
    "object_id": {"dtype": "int", "nullable": False},
    "box_x1": {"dtype": "float", "nullable": False},
    "box_y1": {"dtype": "float", "nullable": False},
    "box_x2": {"dtype": "float", "nullable": False},
    "box_y2": {"dtype": "float", "nullable": False},
    "detector_score": {"dtype": "float", "nullable": False},
    "image_width": {"dtype": "int", "nullable": False},
    "image_height": {"dtype": "int", "nullable": False},
    "rotation_count": {"dtype": "int", "nullable": False},
}


def _covered_area(track: pd.DataFrame) -> int:
    """Count the image pixels covered by at least one of a track's boxes."""
    width = int(track["image_width"].iloc[0])
    height = int(track["image_height"].iloc[0])
    stamped = np.zeros((height, width), dtype=bool)

    corners = track[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(dtype=float)
    for x1, y1, x2, y2 in corners:
        left, top = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        right, bottom = max(int(np.ceil(x2)), 0), max(int(np.ceil(y2)), 0)
        stamped[top:bottom, left:right] = True

    return int(stamped.sum())


class BoxTracks:
    """An immutable table of tracked detection boxes in original image pixels."""

    def __init__(self, df: pd.DataFrame):
        self._df = _validate_dataframe(df.copy(), BOX_TRACK_COLUMNS, "BoxTracks")
        self._validate_geometry()

    @property
    def df(self) -> pd.DataFrame:
        """Return a copy of the tracked-box table."""
        return self._df.copy()

    def _validate_geometry(self) -> None:
        finite_columns = [
            "frame_time",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
            "detector_score",
        ]
        values = self._df[finite_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("BoxTracks validation failed: numeric values must be finite")

        invalid = (
            (self._df["box_x2"] <= self._df["box_x1"])
            | (self._df["box_y2"] <= self._df["box_y1"])
            | (self._df["image_width"] <= 0)
            | (self._df["image_height"] <= 0)
        )
        if invalid.any():
            raise ValueError("BoxTracks validation failed: boxes and source image dimensions must have positive area")

        for key_columns in (["cam_id", "frame_index", "track_id"], ["cam_id", "frame_index", "object_id"]):
            duplicates = self._df.duplicated(subset=key_columns)
            if duplicates.any():
                raise ValueError(
                    f"BoxTracks validation failed: duplicate ({', '.join(key_columns)}) observations are not allowed"
                )

        object_ids_per_track = self._df.groupby(["cam_id", "track_id"])["object_id"].nunique()
        split_tracks = object_ids_per_track[object_ids_per_track > 1]
        if not split_tracks.empty:
            shown = ", ".join(str(camera_track) for camera_track in split_tracks.index)
            raise ValueError(
                "BoxTracks validation failed: each (cam_id, track_id) needs a single object_id; "
                f"found several for {shown}"
            )

        unlinked = self._df[self._df["object_id"] < 0]
        tracks_per_unlinked_id = unlinked.drop_duplicates(["cam_id", "track_id"]).groupby("object_id").size()
        shared_ids = tracks_per_unlinked_id[tracks_per_unlinked_id > 1]
        if not shared_ids.empty:
            shown = ", ".join(str(object_id) for object_id in shared_ids.index)
            raise ValueError(
                "BoxTracks validation failed: each negative object_id must belong to a single camera track; "
                f"found {shown}"
            )

    @classmethod
    def from_csv(cls, path: str | Path) -> BoxTracks:
        """Load tracked boxes saved by :meth:`to_csv`."""
        return cls(pd.read_csv(path))

    def to_csv(self, path: str | Path) -> None:
        """Save tracked boxes with Caliscope's atomic CSV persistence convention."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _safe_write_csv(self._df, path, index=False, float_format=CSV_FLOAT_PRECISION)
        except Exception as e:
            raise PersistenceError(f"Failed to save box tracks to {path}: {e}") from e

    def filter_to_objects(self, object_ids: Iterable[int]) -> BoxTracks:
        """Return a copy containing only rows whose object_id is in object_ids.

        Mirrors ImagePoints.filter_to_objects: absent IDs are not an error, and
        dropped object IDs are logged at info level.
        """
        valid = set(object_ids)
        mask = self._df["object_id"].isin(valid)
        dropped = self._df[~mask]
        if len(dropped) > 0:
            for oid in sorted(dropped["object_id"].unique()):
                n = int((dropped["object_id"] == oid).sum())
                logger.info(f"filter_to_objects: dropped object_id={oid} ({n} rows)")
        return BoxTracks(self._df[mask].copy())

    def exclude_edge_boxes(self, margin_px: float) -> BoxTracks:
        """Remove boxes touching an inclusive source-image edge margin."""
        if not np.isfinite(margin_px) or margin_px < 0:
            raise ValueError("margin_px must be finite and non-negative")

        keep = (
            (self._df["box_x1"] > margin_px)
            & (self._df["box_y1"] > margin_px)
            & (self._df["box_x2"] < self._df["image_width"] - margin_px)
            & (self._df["box_y2"] < self._df["image_height"] - margin_px)
        )
        return BoxTracks(self._df[keep].copy())
