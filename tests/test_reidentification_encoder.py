"""Preprocessing and normalization checks for the re-identification encoder, without ONNX weights."""

from pathlib import Path

import numpy as np
import pytest

from caliscope.trackers.model_card import ReidentificationCard
from caliscope.trackers.reidentification_encoder import ReidentificationEncoder


def _card(model_path: Path = Path("encoder.onnx")) -> ReidentificationCard:
    return ReidentificationCard(
        name="test encoder",
        model_path=model_path,
        input_width=8,
        input_height=16,
        pixel_mean=(0.5, 0.5, 0.5),
        pixel_std=(0.5, 0.5, 0.25),
    )


class _Session:
    """Returns each item's per-channel mean, so the output reflects the preprocessed input."""

    def __init__(self):
        self.batches: list[np.ndarray] = []

    def run(self, _output_names, inputs):
        batch = inputs["input"]
        self.batches.append(batch)
        return [batch.mean(axis=(2, 3))]


def _encoder() -> tuple[ReidentificationEncoder, _Session]:
    encoder = ReidentificationEncoder.__new__(ReidentificationEncoder)
    encoder.card = _card()
    session = _Session()
    setattr(encoder, "session", session)  # a stand-in for the onnxruntime session
    encoder.input_name = "input"
    return encoder, session


def _solid(bgr: tuple[int, int, int], height: int = 30, width: int = 12) -> np.ndarray:
    return np.full((height, width, 3), bgr, dtype=np.uint8)


def test_solid_bgr_crop_arrives_as_normalized_rgb_nchw():
    encoder, session = _encoder()

    encoder.encode([_solid((0, 51, 255))])

    batch = session.batches[0]
    assert batch.shape == (1, 3, 16, 8)
    assert batch.dtype == np.float32
    expected_rgb = ((1.0 - 0.5) / 0.5, (0.2 - 0.5) / 0.5, (0.0 - 0.5) / 0.25)
    for channel, value in enumerate(expected_rgb):
        np.testing.assert_allclose(batch[0, channel], value, atol=1e-6)


def test_embeddings_are_unit_rows_in_input_order_from_one_crop_per_call():
    encoder, session = _encoder()
    colours = [(0, 0, 255), (255, 0, 0)] * 3
    crops = [_solid(colour, height=10 + i) for i, colour in enumerate(colours)]

    embeddings = encoder.encode(crops)

    assert [len(batch) for batch in session.batches] == [1] * len(crops)
    assert embeddings.shape == (len(crops), 3)
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)
    red_first = embeddings[0, 0] > embeddings[0, 2]
    blue_second = embeddings[1, 2] > embeddings[1, 0]
    assert red_first and blue_second
    np.testing.assert_allclose(embeddings[0::2], np.tile(embeddings[0], (len(crops) // 2, 1)), atol=1e-6)
    np.testing.assert_allclose(embeddings[1::2], np.tile(embeddings[1], (len(crops) // 2, 1)), atol=1e-6)


def test_missing_model_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Re-identification model not found"):
        ReidentificationEncoder(_card(tmp_path / "absent.onnx"))
