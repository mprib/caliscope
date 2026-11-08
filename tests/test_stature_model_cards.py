from dataclasses import replace
from pathlib import Path

import pytest

from caliscope.trackers.model_card import BUILTIN_CARDS_DIR, ModelCard, StatureChain


@pytest.mark.parametrize("path", sorted(BUILTIN_CARDS_DIR.glob("rtmpose_*_halpe26.toml")))
def test_halpe26_cards_define_the_stature_chain(path: Path):
    cue = ModelCard.from_toml(path).stature_cue(meters=1.78)

    assert cue.left_path == (24, 15, 13, 11, 5)
    assert cue.right_path == (25, 16, 14, 12, 6)
    assert cue.shoulder_ids == (5, 6)
    assert cue.head == 17
    assert cue.meters == 1.78


def test_card_without_stature_chain_raises_clearly():
    card = ModelCard.from_toml(BUILTIN_CARDS_DIR / "rtmpose_m_coco_wholebody.toml")

    with pytest.raises(ValueError, match="does not define a stature chain"):
        card.stature_cue(meters=1.78)


def test_card_stature_chain_rejects_unknown_point_name():
    card = ModelCard.from_toml(BUILTIN_CARDS_DIR / "rtmpose_m_halpe26.toml")
    invalid = replace(
        card,
        stature=StatureChain(
            ("left_heel", "unknown"), ("right_heel", "right_shoulder"), ("left_shoulder", "right_shoulder"), "head"
        ),
    )

    with pytest.raises(ValueError, match="unknown keypoints"):
        invalid.stature_cue(meters=1.78)
