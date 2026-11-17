"""Checks for the box-limited re-identification embedding pass."""

import numpy as np
import pandas as pd

from caliscope.core import object_tracking
from caliscope.core.box_tracks import BOX_TRACK_COLUMNS, BoxTracks
from caliscope.core.object_tracking import extract_reidentification_embeddings


def _row(*, track_id: int, rotation_count: int = 0, box_x2: float = 40.0) -> dict:
    return {
        "sync_index": 3,
        "cam_id": 0,
        "frame_index": 7,
        "frame_time": 1.2,
        "track_id": track_id,
        "object_id": -track_id,
        "box_x1": 10.0,
        "box_y1": 20.0,
        "box_x2": box_x2,
        "box_y2": 80.0,
        "detector_score": 0.9,
        "image_width": 100,
        "image_height": 120,
        "rotation_count": rotation_count,
    }


class _Source:
    def __init__(self):
        self._returned = False

    @classmethod
    def from_path(cls, *args, **kwargs):
        return cls()

    def next_frame(self):
        if self._returned:
            return None
        self._returned = True
        return type("Raw", (), {"frame_index": 7, "frame": np.zeros((120, 100, 3), dtype=np.uint8)})()

    def close(self):
        pass


class _Encoder:
    """Embeds each crop as its normalized (height, width), and records the crops it saw."""

    def __init__(self):
        self.calls: list[list[tuple[int, ...]]] = []

    def encode(self, crops):
        self.calls.append([crop.shape for crop in crops])
        sizes = np.array([crop.shape[:2] for crop in crops], dtype=np.float32)
        return sizes / np.linalg.norm(sizes, axis=1, keepdims=True)


def test_one_embedding_per_box_cut_from_the_rotated_frame(tmp_path, monkeypatch):
    """Each box is cropped from its own inference rotation; a sliver box is skipped."""
    rows = [
        _row(track_id=1),
        _row(track_id=2, rotation_count=1),
        _row(track_id=3, box_x2=11.0),
    ]
    video = tmp_path / "cam_0.mp4"
    video.touch()
    monkeypatch.setattr(object_tracking, "FrameSource", _Source)
    encoder = _Encoder()

    embeddings = extract_reidentification_embeddings(
        {0: video}, encoder, BoxTracks(pd.DataFrame(rows, columns=list(BOX_TRACK_COLUMNS)))
    )

    assert encoder.calls == [[(60, 30, 3), (30, 60, 3)]]
    assert embeddings.index.values.tolist() == [[3, 0, 1], [3, 0, 2]]
    np.testing.assert_allclose(embeddings.vectors[0], np.array([60, 30]) / np.hypot(60, 30), atol=1e-6)
    np.testing.assert_allclose(embeddings.vectors[1], np.array([30, 60]) / np.hypot(60, 30), atol=1e-6)
