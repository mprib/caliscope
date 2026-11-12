"""Validation and persistence checks for ReidentificationEmbeddings."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caliscope.core.reidentification_embeddings import ReidentificationEmbeddings


def _index(keys: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(keys, columns=["sync_index", "cam_id", "track_id"])


def test_construction_sorts_rows_and_keeps_vectors_aligned():
    keys = [(5, 1, 2), (3, 0, 1), (5, 0, 7)]
    vectors = np.eye(3, dtype=np.float32)

    embeddings = ReidentificationEmbeddings(_index(keys), vectors)

    assert embeddings.index.values.tolist() == [[3, 0, 1], [5, 0, 7], [5, 1, 2]]
    np.testing.assert_array_equal(embeddings.vectors, vectors[[1, 2, 0]])
    assert not embeddings.vectors.flags.writeable
    assert len(embeddings) == 3


def test_duplicate_key_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ReidentificationEmbeddings(_index([(3, 0, 1), (3, 0, 1)]), np.eye(2, dtype=np.float32))


def test_non_unit_row_is_rejected():
    vectors = np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="unit length"):
        ReidentificationEmbeddings(_index([(3, 0, 1), (4, 0, 1)]), vectors)


def test_npz_round_trip(tmp_path: Path):
    vectors = np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    original = ReidentificationEmbeddings(_index([(4, 2, 9), (3, 1, 1)]), vectors)
    path = tmp_path / "nested" / "embeddings.npz"

    original.to_npz(path)
    restored = ReidentificationEmbeddings.from_npz(path)

    pd.testing.assert_frame_equal(restored.index, original.index)
    np.testing.assert_array_equal(restored.vectors, original.vectors)
