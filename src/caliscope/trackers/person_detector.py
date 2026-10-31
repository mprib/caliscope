"""YOLOX person detector that tells a top-down pose model where to look.

The detector returns all person boxes for a frame; largest_person selects one
for the generic frame-local pose path. Recording-level box association and
primary-subject selection belong to the separate box-first extraction stages.

The ONNX file is Megvii's YOLOX export, which emits the raw prediction grid
(strides 8, 16 and 32 over the letterboxed input). Each row is cx, cy, log w,
log h in grid units, objectness, then one score per COCO class. Decoding and
non-maximum suppression happen here.
"""

import logging

import cv2
import numpy as np

from caliscope.trackers.model_card import DetectorCard

logger = logging.getLogger(__name__)

STRIDES = (8, 16, 32)
PAD_VALUE = 114  # gray used for letterbox padding in YOLOX training


class PersonDetector:
    """Find person boxes in a BGR frame with a YOLOX ONNX model."""

    def __init__(self, card: DetectorCard) -> None:
        if not card.onnx_exists:
            raise FileNotFoundError(f"Detector model not found: {card.model_path}")

        from caliscope.onnx_session import create_inference_session

        self.card = card
        logger.info(f"Loading detector model: {card.model_path}")
        self.session = create_inference_session(card.model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Grid cell offsets and strides for every row of the model output
        offsets, strides = [], []
        for stride in STRIDES:
            columns, rows = card.input_width // stride, card.input_height // stride
            xv, yv = np.meshgrid(np.arange(columns), np.arange(rows))
            offsets.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
            strides.append(np.full((rows * columns, 1), stride))
        self._grid_offsets = np.concatenate(offsets).astype(np.float32)
        self._grid_strides = np.concatenate(strides).astype(np.float32)

    def people(self, frame: np.ndarray, min_score: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return every person box in a BGR frame and its score.

        Boxes are (N, 4) as x1, y1, x2, y2 in frame pixels, clamped to the frame;
        scores are (N,). min_score defaults to the card's score_threshold. A
        tracker passes a lower value so it can use weak detections to keep an
        existing track alive.

        The frame is letterboxed into the model input: scaled to fit, placed at
        the top-left, and padded with gray. The model takes raw BGR values in
        0-255 with no normalisation.
        """
        if min_score is None:
            min_score = self.card.score_threshold
        frame_h, frame_w = frame.shape[:2]
        scale = min(self.card.input_width / frame_w, self.card.input_height / frame_h)
        scaled_w, scaled_h = int(frame_w * scale), int(frame_h * scale)
        canvas = np.full((self.card.input_height, self.card.input_width, 3), PAD_VALUE, dtype=np.uint8)
        canvas[:scaled_h, :scaled_w] = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        model_input = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)

        predictions = np.asarray(self.session.run(None, {self.input_name: model_input})[0])[0]

        scores = predictions[:, 4] * predictions[:, 5 + self.card.class_id]
        keep = scores >= min_score
        if not np.any(keep):
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
        centres = (predictions[keep, :2] + self._grid_offsets[keep]) * self._grid_strides[keep] / scale
        sizes = np.exp(predictions[keep, 2:4]) * self._grid_strides[keep] / scale
        scores = scores[keep]

        # Overlapping predictions of the same person collapse to the highest-scoring one
        boxes_xywh = np.column_stack((centres - sizes / 2, sizes))
        kept = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), min_score, self.card.nms_threshold)
        kept = np.asarray(kept, dtype=int).reshape(-1)

        boxes = np.column_stack((centres[kept] - sizes[kept] / 2, centres[kept] + sizes[kept] / 2))
        boxes = np.clip(boxes, 0, (frame_w, frame_h, frame_w, frame_h)).astype(np.float32)
        return boxes, scores[kept].astype(np.float32)

    def largest_person(self, frame: np.ndarray) -> tuple[float, float, float, float] | None:
        """Return the largest person box as (x1, y1, x2, y2) in frame pixels, or None."""
        boxes, _ = self.people(frame)
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = boxes[np.argmax(areas)]
        return float(x1), float(y1), float(x2), float(y2)
