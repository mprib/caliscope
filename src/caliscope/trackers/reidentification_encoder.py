"""Person re-identification encoder that turns box crops into embedding vectors.

Each crop is resized, normalized and passed through an ONNX model; each output
row is scaled to unit length. An embedding only supports asking whether two
crops show the same individual. It does not say who anyone is.
"""

import logging
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.trackers.model_card import ReidentificationCard

logger = logging.getLogger(__name__)


class ReidentificationEncoder:
    """Encode BGR crops of single objects as unit-length embedding vectors."""

    def __init__(self, card: ReidentificationCard) -> None:
        if not card.onnx_exists:
            raise FileNotFoundError(f"Re-identification model not found: {card.model_path}")

        from caliscope.onnx_session import create_inference_session

        self.card = card
        logger.info(f"Loading re-identification model: {card.model_path}")
        self.session = create_inference_session(card.model_path)
        self.input_name = self.session.get_inputs()[0].name

    def encode(self, crops: Sequence[np.ndarray]) -> NDArray[np.float32]:
        """Return one unit-length embedding row per crop, in input order.

        Each crop is a BGR uint8 image (H, W, 3) of any size.

        Raises:
            ValueError: if a crop is not 3-channel or has a zero-length side.
        """
        if len(crops) == 0:
            dimension = self.session.get_outputs()[0].shape[-1]
            return np.empty((0, dimension if isinstance(dimension, int) else 0), dtype=np.float32)

        # One crop per call keeps the input shape constant. The CUDA provider re-tunes its convolutions whenever
        # the shape changes, which costs far more than the batching saves.
        embeddings = np.stack(
            [
                np.asarray(self.session.run(None, {self.input_name: self._preprocess(crop)[None]})[0]).reshape(-1)
                for crop in crops
            ]
        ).astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)

    def _preprocess(self, crop: np.ndarray) -> NDArray[np.float32]:
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"Crops must be 3-channel images (H, W, 3), got shape {crop.shape}")
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            raise ValueError(f"Crops must have non-zero height and width, got shape {crop.shape}")
        resized = cv2.resize(crop, (self.card.input_width, self.card.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - np.asarray(self.card.pixel_mean, dtype=np.float32)) / np.asarray(
            self.card.pixel_std, dtype=np.float32
        )
        return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
