"""Parsing checks for re-identification model cards."""

from pathlib import Path

import pytest

from caliscope.trackers.model_card import ReidentificationCard

CARD_TOML = """
[reidentification]
name = "test encoder"
model_path = "encoder.onnx"
input_size = [128, 256]
pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]

[source]
url = "https://example.com/encoder.onnx"
extraction = "direct"
license = "MIT"
"""


def test_card_parses_fields_and_resolves_model_path(tmp_path: Path):
    path = tmp_path / "encoder.toml"
    path.write_text(CARD_TOML)

    card = ReidentificationCard.from_toml(path, models_dir=tmp_path)
    unresolved = ReidentificationCard.from_toml(path)

    assert card.name == "test encoder"
    assert (card.input_width, card.input_height) == (128, 256)
    assert card.pixel_mean == pytest.approx((0.485, 0.456, 0.406))
    assert card.pixel_std == pytest.approx((0.229, 0.224, 0.225))
    assert card.model_path == tmp_path / "encoder.onnx"
    assert unresolved.model_path == Path("encoder.onnx")
    assert card.has_source_url and card.license_info == "MIT"
    assert not card.onnx_exists


def test_card_rejects_missing_pixel_mean(tmp_path: Path):
    path = tmp_path / "encoder.toml"
    path.write_text(CARD_TOML.replace("pixel_mean = [0.485, 0.456, 0.406]\n", ""))

    with pytest.raises(ValueError, match="pixel_mean"):
        ReidentificationCard.from_toml(path)
