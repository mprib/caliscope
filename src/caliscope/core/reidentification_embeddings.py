"""Re-identification embeddings for tracked boxes, keyed by moment, camera and track."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

KEY_COLUMNS = ["sync_index", "cam_id", "track_id"]
# Largest allowed difference between a stored row's L2 norm and 1.
UNIT_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True, eq=False)
class ReidentificationEmbeddings:
    """One unit-length re-identification vector per box.

    index has one row per embedding with its sync_index, cam_id and track_id;
    vectors holds the matching rows. Two vectors are compared by their dot
    product. An embedding supports asking whether two boxes show the same
    individual and carries no identity.
    """

    _index: pd.DataFrame
    _vectors: NDArray[np.float32]

    def __post_init__(self) -> None:
        missing = [column for column in KEY_COLUMNS if column not in self._index.columns]
        if missing:
            raise ValueError(f"ReidentificationEmbeddings validation failed: column(s) {missing} not in index")

        index = pd.DataFrame({column: _integer_column(self._index[column], column) for column in KEY_COLUMNS})
        if index.duplicated().any():
            raise ValueError(
                "ReidentificationEmbeddings validation failed: duplicate (sync_index, cam_id, track_id) rows"
            )

        vectors = np.array(self._vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"ReidentificationEmbeddings validation failed: vectors must be 2-D, got {vectors.ndim}-D")
        if len(vectors) != len(index):
            raise ValueError(
                f"ReidentificationEmbeddings validation failed: {len(vectors)} vectors for {len(index)} index rows"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("ReidentificationEmbeddings validation failed: vectors must be finite")
        norms = np.linalg.norm(vectors, axis=1)
        if (np.abs(norms - 1.0) > UNIT_NORM_TOLERANCE).any():
            raise ValueError("ReidentificationEmbeddings validation failed: every vector must have unit length")

        order = np.lexsort((index["track_id"], index["cam_id"], index["sync_index"]))
        sorted_vectors = vectors[order]
        sorted_vectors.setflags(write=False)
        object.__setattr__(self, "_index", index.iloc[order].reset_index(drop=True))
        object.__setattr__(self, "_vectors", sorted_vectors)

    @property
    def index(self) -> pd.DataFrame:
        """Return a copy of the (sync_index, cam_id, track_id) table, aligned with vectors."""
        return self._index.copy()

    @property
    def vectors(self) -> NDArray[np.float32]:
        """Return the read-only (N, D) array of unit-length embeddings."""
        return self._vectors

    def __len__(self) -> int:
        return len(self._index)

    def to_npz(self, path: str | Path) -> None:
        """Save keys and vectors as a compressed npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            sync_index=self._index["sync_index"].to_numpy(),
            cam_id=self._index["cam_id"].to_numpy(),
            track_id=self._index["track_id"].to_numpy(),
            vectors=self._vectors,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> ReidentificationEmbeddings:
        """Load embeddings saved by :meth:`to_npz`."""
        with np.load(Path(path), allow_pickle=False) as data:
            index = pd.DataFrame({column: data[column] for column in KEY_COLUMNS})
            return cls(index, data["vectors"])


def _integer_column(values: pd.Series, column: str) -> pd.Series:
    as_float = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    if not np.isfinite(as_float).all() or (as_float != np.round(as_float)).any():
        raise ValueError(
            f"ReidentificationEmbeddings validation failed: column '{column}' must hold integers with no nulls"
        )
    return pd.Series(as_float.astype(np.int64), index=values.index)
