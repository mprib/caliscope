"""Validated detection boxes retained between tracking and pose stages."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from caliscope.core.point_data import _validate_dataframe
from caliscope.persistence import CSV_FLOAT_PRECISION, PersistenceError, _safe_write_csv

if TYPE_CHECKING:
    from caliscope.core.track_association import TrackAssociations

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


def _bit_reversed_priorities(moments: list[int]) -> dict[int, int]:
    """Map each sorted moment to its rank with the bits reversed (van der Corput order)."""
    bits = max(len(moments) - 1, 0).bit_length()
    priorities: dict[int, int] = {}
    for rank, moment in enumerate(moments):
        reversed_rank = 0
        for bit in range(bits):
            reversed_rank = (reversed_rank << 1) | ((rank >> bit) & 1)
        priorities[moment] = reversed_rank
    return priorities


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

    def sample_for_association(self, frames_per_track: int = 12) -> BoxTracks:
        """Keep every box at a set of moments chosen so each track is sampled.

        Only moments where at least two cameras have a box are candidates. These
        are ranked by one global coarse-to-fine order, and each camera track
        nominates its frames_per_track best-ranked candidates, or all of them if
        it has fewer. Because the order is shared, tracks of one object in
        different cameras nominate the same moments. Every box at any nominated
        moment is kept, in every camera and track, so frames_per_track is a floor
        per track where enough shared moments exist, not a cap.

        Deterministic, and image position plays no part. Compose with
        exclude_edge_boxes to drop edge boxes first. The result is ordinary
        BoxTracks, empty when no moment is shared between cameras.

        Raises:
            ValueError: if frames_per_track is not an integer >= 1.
        """
        if isinstance(frames_per_track, bool) or not isinstance(frames_per_track, int) or frames_per_track < 1:
            raise ValueError("frames_per_track must be an integer >= 1")

        cameras_per_moment = self._df.groupby("sync_index")["cam_id"].nunique()
        shared_moments = sorted(int(moment) for moment in cameras_per_moment.index[cameras_per_moment >= 2])
        priorities = _bit_reversed_priorities(shared_moments)

        eligible = self._df[self._df["sync_index"].isin(shared_moments)]
        chosen: set[int] = set()
        for _, track in eligible.groupby(["cam_id", "track_id"]):
            track_moments = sorted({int(moment) for moment in track["sync_index"]}, key=priorities.__getitem__)
            chosen.update(track_moments[:frames_per_track])

        return BoxTracks(self._df[self._df["sync_index"].isin(chosen)].copy())

    def apply_associations(self, associations: TrackAssociations, keep_unlinked: bool = False) -> BoxTracks:
        """Write linked object IDs onto the boxes of each associated track.

        Rows of tracks absent from associations.linked are dropped, or kept with
        their current object_id when keep_unlinked is True.

        Raises:
            ValueError: if associations name a (cam_id, track_id) absent from these
                boxes, or if two tracks mapped to one object overlap in time in one
                camera (raised by BoxTracks validation).
        """
        links = associations.linked[["cam_id", "track_id", "object_id"]].rename(
            columns={"object_id": "linked_object_id"}
        )
        present = set(zip(self._df["cam_id"].tolist(), self._df["track_id"].tolist()))
        absent = sorted(
            (int(cam_id), int(track_id))
            for cam_id, track_id in zip(links["cam_id"].tolist(), links["track_id"].tolist())
            if (cam_id, track_id) not in present
        )
        if absent:
            raise ValueError(f"associations name (cam_id, track_id) {absent} absent from these boxes")

        merged = self._df.merge(links, on=["cam_id", "track_id"], how="left", validate="many_to_one")
        is_linked = merged["linked_object_id"].notna()
        if not keep_unlinked:
            merged = merged[is_linked]
        merged = merged.assign(object_id=merged["linked_object_id"].fillna(merged["object_id"]))
        return BoxTracks(merged.drop(columns=["linked_object_id"]))
