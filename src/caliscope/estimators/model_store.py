"""Estimator model specs and on-demand download.

Estimator ONNX models (MoGe, GeoCalib fields) don't fit the tracker
``ModelCard`` schema (no keypoints, no SimCC/heatmap format, dynamic
resolution), so each estimator declares a frozen ``EstimatorModelSpec``
constant instead of a TOML card. Weights live in the same ``MODELS_DIR``
as tracker models; the tracker registry ignores them because it scans
TOML cards, not ``.onnx`` files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from caliscope import MODELS_DIR
from caliscope.trackers.model_download import download_and_extract_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EstimatorModelSpec:
    """Downloadable ONNX model for an estimator. Satisfies ``ModelSource``."""

    name: str
    filename: str
    source_url: str | None
    sha256: str | None
    extraction: str | None = "direct"
    file_size_mb: float | None = None
    license_info: str | None = None
    model_path: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", MODELS_DIR / self.filename)


def ensure_model(spec: EstimatorModelSpec) -> Path:
    """Return the local path to the model weights, downloading on first use."""
    if spec.model_path.exists():
        return spec.model_path
    logger.info(f"Estimator model '{spec.name}' not found locally; downloading.")
    return download_and_extract_model(spec, MODELS_DIR)
