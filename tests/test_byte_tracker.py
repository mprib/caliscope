from __future__ import annotations

import numpy as np

from caliscope.trackers.byte_tracker import ByteTracker


def test_update_returns_detector_box_and_score_for_high_and_low_matches():
    tracker = ByteTracker()
    box = np.array([[10.0, 20.0, 30.0, 60.0]])

    first = tracker.update(box, np.array([0.9]))
    second = tracker.update(box, np.array([0.2]))

    first_id, first_box, first_score = first[0]
    second_id, second_box, second_score = second[0]
    assert first_id == second_id == 1
    np.testing.assert_array_equal(first_box, box[0])
    np.testing.assert_array_equal(second_box, box[0])
    assert first_score == 0.9
    assert second_score == 0.2
