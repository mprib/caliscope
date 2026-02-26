"""Integration test for OnnxTracker using a tiny constant-output fixture model.

The fixture model (tests/fixtures/onnx/simcc_3pt.onnx) always outputs the
same SimCC logit peaks regardless of input, producing deterministic keypoints.
This exercises the full pipeline: ModelCard loading, session creation, preprocessing,
inference, SimCC decoding, confidence thresholding, and coordinate remapping.

Expected frame coordinates for a 640x480 input (see generate_simcc_3pt.py):
    nose:      (320.0, 240.0)  — frame center
    left_eye:  (200.0, 160.0)
    right_eye: (440.0, 160.0)
    low_conf:  (66.7, 120.0)   — filtered out (confidence ~0.1 < 0.3 threshold)
"""

from pathlib import Path

import numpy as np

from caliscope.trackers.model_card import ModelCard
from caliscope.trackers.onnx_tracker import OnnxTracker

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "onnx"

EXPECTED_COORDS = np.array(
    [
        [320.0, 240.0],  # nose: simcc_x[48]/2=24.0, (24.0-0)/0.075=320; simcc_y[64]/2=32.0, (32.0-14)/0.075=240
        [200.0, 160.0],  # left_eye: simcc_x[30]/2=15.0, 15.0/0.075=200; simcc_y[52]/2=26.0, (26.0-14)/0.075=160
        [440.0, 160.0],  # right_eye: simcc_x[66]/2=33.0, 33.0/0.075=440; simcc_y[52]/2=26.0, (26.0-14)/0.075=160
    ],
    dtype=np.float32,
)


def _load_tracker() -> OnnxTracker:
    card = ModelCard.from_toml(FIXTURE_DIR / "simcc_3pt.toml", models_dir=FIXTURE_DIR)
    return OnnxTracker(card)


def test_onnx_tracker_loads_from_card():
    """ModelCard + OnnxTracker construction succeeds with the fixture model."""
    tracker = _load_tracker()
    assert tracker.card.name == "Test Pose 3pt"
    assert tracker.card.format == "simcc"
    assert len(tracker.card.point_name_to_id) == 4


def test_onnx_tracker_detects_all_keypoints():
    """Full pipeline produces a PointPacket with all 3 keypoints."""
    tracker = _load_tracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = tracker.get_points(frame)

    assert len(packet.point_id) == 3
    assert np.array_equal(packet.point_id, np.array([0, 1, 2], dtype=np.int32))
    assert packet.img_loc.shape == (3, 2)
    assert packet.confidence is not None
    assert all(packet.confidence >= 0.3)


def test_onnx_tracker_coordinate_accuracy():
    """Decoded coordinates match expected frame positions within 1 pixel."""
    tracker = _load_tracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = tracker.get_points(frame)

    np.testing.assert_allclose(packet.img_loc, EXPECTED_COORDS, atol=1.0)


def test_onnx_tracker_point_names():
    """Point name mapping round-trips through the model card."""
    tracker = _load_tracker()
    assert tracker.get_point_name(0) == "nose"
    assert tracker.get_point_name(1) == "left_eye"
    assert tracker.get_point_name(2) == "right_eye"
    assert tracker.get_point_name(3) == "low_conf"


def test_onnx_tracker_filters_low_confidence():
    """Keypoints below confidence_threshold are excluded from PointPacket."""
    tracker = _load_tracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = tracker.get_points(frame)

    # Model has 4 keypoints, but low_conf (id=3) has logit 0.1 < threshold 0.3
    assert len(packet.point_id) == 3
    assert 3 not in packet.point_id


def test_onnx_tracker_wireframe():
    """Wireframe segments are parsed from the TOML card."""
    tracker = _load_tracker()
    wf = tracker.card.wireframe
    assert wf is not None
    assert len(wf.segments) == 1
    assert wf.segments[0].name == "eyes"
    assert wf.segments[0].point_A == "left_eye"
    assert wf.segments[0].point_B == "right_eye"


if __name__ == "__main__":
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    tracker = _load_tracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = tracker.get_points(frame)

    print(f"Detected {len(packet.point_id)} keypoints:")
    for i, pid in enumerate(packet.point_id):
        name = tracker.get_point_name(int(pid))
        x, y = packet.img_loc[i]
        conf = packet.confidence[i] if packet.confidence is not None else None
        print(f"  {name:>12s} (id={pid}): ({x:7.1f}, {y:7.1f})  conf={conf}")

    print("\nExpected:")
    for name, (ex, ey) in zip(["nose", "left_eye", "right_eye"], EXPECTED_COORDS):
        print(f"  {name:>12s}:          ({ex:7.1f}, {ey:7.1f})")
