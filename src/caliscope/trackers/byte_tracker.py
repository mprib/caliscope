"""ByteTrack-style tracking of person boxes within one camera.

Adapted from Monokin's `byte_tracker.py`, which adapts GEM-X
`gem/utils/yolox_detector.py` at commit 953b3871c15a38acec6b7cccec18ed6019b4512b.
GEM-X is Apache-2.0 and ByteTrack (Zhang et al., ECCV 2022) is MIT.
Caliscope changes: boxes and scores are plain arrays rather than value objects,
and the tracker returns track IDs for the boxes it was given.

Each frame, high-score boxes are matched to existing tracks by IoU, then
low-score boxes are tried on the tracks still unmatched, so a briefly weak
detection keeps its track. New tracks start only from high-score boxes. A track
unmatched for more than max_lost frames ends; a person who returns after that
gets a new ID. Track IDs are local to one tracker, which serves one camera.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

LOW_SCORE_THRESHOLD = 0.1


class ByteTracker:
    """Assign track IDs to one camera's person boxes, frame by frame in order."""

    def __init__(self, high_threshold: float = 0.5, match_threshold: float = 0.3, max_lost: int = 30) -> None:
        self.high_threshold = high_threshold
        self.match_threshold = match_threshold
        self.max_lost = max_lost
        self.tracks: list[_KalmanBoxTrack] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Process one frame's boxes (N, 4 as x1, y1, x2, y2) and scores (N,).

        Returns (track_id, box) for every track matched in this frame. The box is
        the detector's box, not the filtered estimate, so a caller can crop it.
        """
        for track in self.tracks:
            track.predict()

        high = scores >= self.high_threshold
        low = (scores > LOW_SCORE_THRESHOLD) & ~high
        high_boxes, high_scores = boxes[high], scores[high]
        low_boxes, low_scores = boxes[low], scores[low]

        matched: list[tuple[int, np.ndarray]] = []
        pairs, unmatched_tracks, unmatched_boxes = _match(self.tracks, high_boxes, self.match_threshold)
        for track_index, box_index in pairs:
            self.tracks[track_index].update(high_boxes[box_index], float(high_scores[box_index]))
            matched.append((self.tracks[track_index].track_id, high_boxes[box_index]))

        remaining = [self.tracks[i] for i in unmatched_tracks]
        low_pairs, _, _ = _match(remaining, low_boxes, self.match_threshold)
        for track_index, box_index in low_pairs:
            remaining[track_index].update(low_boxes[box_index], float(low_scores[box_index]))
            matched.append((remaining[track_index].track_id, low_boxes[box_index]))

        for box_index in unmatched_boxes:
            track = _KalmanBoxTrack(high_boxes[box_index], float(high_scores[box_index]), self._next_id)
            self._next_id += 1
            self.tracks.append(track)
            matched.append((track.track_id, high_boxes[box_index]))

        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_lost]
        return matched


def _match(
    tracks: list["_KalmanBoxTrack"], boxes: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match tracks to boxes by IoU; return pairs, unmatched track and box indices."""
    if len(tracks) == 0 or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))

    iou = _iou_batch(np.array([track.box for track in tracks]), boxes)
    rows, columns = linear_sum_assignment(1.0 - iou)
    pairs = [(int(r), int(c)) for r, c in zip(rows, columns) if iou[r, c] >= threshold]
    matched_tracks = {r for r, _ in pairs}
    matched_boxes = {c for _, c in pairs}
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_boxes = [i for i in range(len(boxes)) if i not in matched_boxes]
    return pairs, unmatched_tracks, unmatched_boxes


class _KalmanBoxTrack:
    """One track: a constant-velocity Kalman filter on box centre, aspect ratio and height."""

    def __init__(self, box: np.ndarray, score: float, track_id: int) -> None:
        self.x = np.concatenate([_xyxy_to_xyah(box), np.zeros(4)])
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01
        self.R = np.diag([1.0, 1.0, 10.0, 1.0])
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.track_id = track_id
        self.time_since_update = 0
        self.score = score

    @property
    def box(self) -> np.ndarray:
        return _xyah_to_xyxy(self.x[:4])

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1

    def update(self, box: np.ndarray, score: float) -> None:
        residual = _xyxy_to_xyah(box) - self.H @ self.x
        gain = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + gain @ residual
        self.P = (np.eye(8) - gain @ self.H) @ self.P
        self.time_since_update = 0
        self.score = score


def _xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float64)
    width, height = x2 - x1, y2 - y1
    return np.array([x1 + width / 2, y1 + height / 2, width / (height + 1e-7), height])


def _xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    centre_x, centre_y, aspect_ratio, height = state
    width = aspect_ratio * height
    return np.array([centre_x - width / 2, centre_y - height / 2, centre_x + width / 2, centre_y + height / 2])


def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise intersection over union of two xyxy box arrays."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + 1e-7)
