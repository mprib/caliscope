"""Unit tests for letterbox-boundary filtering in the YOLOX person detector."""

from pathlib import Path

import numpy as np

from caliscope.trackers.person_detector import PersonDetector, STRIDES
from caliscope.trackers.model_card import DetectorCard


class _Session:
    def __init__(self, predictions: np.ndarray):
        self.predictions = predictions

    def run(self, _output_names, _inputs):
        return [self.predictions[None]]


def _detector(predictions: np.ndarray) -> PersonDetector:
    """Make a 640-square detector without loading ONNX weights."""
    detector = PersonDetector.__new__(PersonDetector)
    detector.card = DetectorCard(
        name="test detector",
        model_path=Path("test.onnx"),
        format="yolox",
        input_width=640,
        input_height=640,
        class_id=0,
        score_threshold=0.5,
        nms_threshold=0.45,
    )
    detector.session = _Session(predictions)
    detector.input_name = "images"

    offsets, strides = [], []
    for stride in STRIDES:
        columns = rows = 640 // stride
        xv, yv = np.meshgrid(np.arange(columns), np.arange(rows))
        offsets.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        strides.append(np.full((rows * columns, 1), stride))
    detector._grid_offsets = np.concatenate(offsets).astype(np.float32)
    detector._grid_strides = np.concatenate(strides).astype(np.float32)
    return detector


def _predictions(*detections: tuple[float, float, float]) -> np.ndarray:
    """Build raw 640-grid outputs for detections in the stride-16 top row.

    Each detection is (centre_y, score, width); height is fixed at 20 pixels.
    """
    predictions = np.zeros((8400, 85), dtype=np.float32)
    for offset, (centre_y, score, width) in enumerate(detections):
        row = 6400 + offset  # First rows of the stride-16 grid, whose y offset is zero.
        predictions[row, :4] = (20 - offset, centre_y / 16, np.log(width / 16), np.log(20 / 16))
        predictions[row, 4:6] = (score, 1.0)
    return predictions


def test_people_discards_box_entirely_in_letterbox_padding():
    detector = _detector(_predictions((560, 0.9, 100)))

    boxes, scores = detector.people(np.zeros((480, 640, 3), dtype=np.uint8))

    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert boxes.dtype == np.float32
    assert scores.dtype == np.float32


def test_people_keeps_partial_edge_box_and_its_score():
    detector = _detector(_predictions((560, 0.95, 100), (475, 0.8, 100)))

    boxes, scores = detector.people(np.zeros((480, 640, 3), dtype=np.uint8))

    np.testing.assert_allclose(boxes, [[270, 465, 370, 480]])
    np.testing.assert_allclose(scores, [0.8])
