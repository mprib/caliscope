"""Focused checks for box-first object tracking helpers and boundaries."""

import numpy as np
import pandas as pd
import pytest

from caliscope.api import track_objects as public_track_objects
from caliscope.core import object_tracking
from caliscope.core.box_tracks import BoxTracks
from caliscope.core.object_tracking import (
    _empty_image_points,
    _rotate_box,
    _unrotate_keypoints,
    _unrotate_box,
    estimate_keypoints,
)
from caliscope.recording import synchronized_timestamps
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps


def _box_tracks(rows: list[dict]) -> BoxTracks:
    return BoxTracks(
        pd.DataFrame(
            rows,
            columns=[
                "sync_index",
                "cam_id",
                "frame_index",
                "frame_time",
                "track_id",
                "object_id",
                "box_x1",
                "box_y1",
                "box_x2",
                "box_y2",
                "detector_score",
                "image_width",
                "image_height",
                "rotation_count",
            ],
        )
    )


def _row(*, cam_id: int = 0, track_id: int = 1, object_id: int = -1) -> dict:
    return {
        "sync_index": 3,
        "cam_id": cam_id,
        "frame_index": 7,
        "frame_time": 1.2,
        "track_id": track_id,
        "object_id": object_id,
        "box_x1": 10.0,
        "box_y1": 20.0,
        "box_x2": 40.0,
        "box_y2": 80.0,
        "detector_score": 0.9,
        "image_width": 100,
        "image_height": 120,
        "rotation_count": 0,
    }


@pytest.mark.parametrize("rotation_count", [0, 1, 2, -1, 3, -3])
def test_box_rotation_round_trips(rotation_count: int):
    """Persisted source boxes survive the detector's inference rotation."""
    original = (10.0, 20.0, 40.0, 80.0)
    rotated = _rotate_box(original, rotation_count, 100, 120)
    restored = _unrotate_box(np.asarray(rotated), rotation_count, 100, 120)

    assert restored == pytest.approx(original)


def test_empty_keypoints_keeps_schema_and_provenance_columns():
    points = _empty_image_points().df

    assert points.empty
    assert {"frame_index", "frame_time", "confidence", "track_id"} <= set(points.columns)


def test_keypoint_estimation_runs_once_per_object_box_in_a_frame(tmp_path, monkeypatch):
    """Two objects in one camera and frame each get keypoints with their own identity."""

    class _Source:
        def __init__(self):
            self._returned = False

        @classmethod
        def from_path(cls, *args, **kwargs):
            return cls()

        def next_frame(self):
            if self._returned:
                return None
            self._returned = True
            return type("Raw", (), {"frame_index": 7, "frame": np.zeros((120, 100, 3), dtype=np.uint8)})()

        def close(self):
            pass

    class _KeypointTracker:
        pixel_format = None

        def __init__(self):
            self.boxes = []

        def keypoints_in_box(self, frame, box):
            self.boxes.append(box)
            centre = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]
            return np.array(centre, dtype=np.float32), np.array([0.9], dtype=np.float32)

    first = _row(track_id=1, object_id=0)
    second = _row(track_id=4, object_id=1) | {"box_x1": 50.0, "box_x2": 90.0}
    video = tmp_path / "cam_0.mp4"
    video.touch()
    monkeypatch.setattr(object_tracking, "FrameSource", _Source)
    keypoint_tracker = _KeypointTracker()

    points = estimate_keypoints(
        {0: video}, keypoint_tracker, _box_tracks([first, second]), min_confidence=0.5
    ).df.sort_values("object_id")

    assert len(keypoint_tracker.boxes) == 2
    assert points[["object_id", "track_id"]].values.tolist() == [[0, 1], [1, 4]]
    assert points["img_loc_x"].tolist() == pytest.approx([25.0, 70.0])


def test_track_objects_numbers_unlinked_objects_by_camera_and_track(tmp_path, monkeypatch):
    """Negative IDs follow sorted (cam_id, track_id), whatever order cameras are given or finish in."""

    class _Source:
        def __init__(self, wanted_indices):
            self._frames = iter(sorted(wanted_indices))

        @classmethod
        def from_path(cls, *_args, wanted_indices, **_kwargs):
            return cls(wanted_indices)

        def next_frame(self):
            try:
                frame_index = next(self._frames)
            except StopIteration:
                return None
            return type("Raw", (), {"frame_index": frame_index, "frame": np.zeros((20, 30, 3), dtype=np.uint8)})()

        def close(self):
            pass

    class _Detector:
        def detect(self, frame, min_score=None):
            boxes = np.array([[1, 2, 10, 12], [15, 2, 25, 12]], dtype=np.float32)
            return boxes, np.array([0.9, 0.8], dtype=np.float32)

    videos = {5: tmp_path / "cam_5.mp4", 2: tmp_path / "cam_2.mp4"}
    for video in videos.values():
        video.touch()
    monkeypatch.setattr(synchronized_timestamps, "read_video_properties", lambda _path: {"fps": 10.0, "frame_count": 1})
    monkeypatch.setattr(object_tracking, "FrameSource", _Source)
    timestamps = SynchronizedTimestamps.from_anchor_frames(videos, {5: 0, 2: 0})

    tracks = object_tracking.track_objects(videos, _Detector(), timestamps=timestamps).df

    identities = tracks[["cam_id", "track_id", "object_id"]].sort_values(["cam_id", "track_id"])
    assert identities.values.tolist() == [[2, 1, -1], [2, 2, -2], [5, 1, -3], [5, 2, -4]]


def test_keypoint_estimation_reports_table_camera_absent_from_videos():
    tracks = _box_tracks([_row(cam_id=5)])

    with pytest.raises(ValueError, match="absent from videos"):
        estimate_keypoints({}, object(), tracks, min_confidence=0.5)


@pytest.mark.parametrize("min_confidence", [-0.01, np.nan, np.inf])
def test_keypoint_estimation_requires_a_finite_nonnegative_cutoff(min_confidence: float):
    tracks = _box_tracks([])

    with pytest.raises(ValueError, match="finite and non-negative"):
        estimate_keypoints({}, object(), tracks, min_confidence=min_confidence)


def test_track_objects_records_source_coordinates_and_frame_metadata(tmp_path, monkeypatch):
    """Detection uses rotated pixels but the persisted record uses source pixels."""

    class _Source:
        def __init__(self):
            self._returned = False

        @classmethod
        def from_path(cls, *args, **kwargs):
            return cls()

        def next_frame(self):
            if self._returned:
                return None
            self._returned = True
            return type("Raw", (), {"frame_index": 4, "frame": np.zeros((120, 100, 3), dtype=np.uint8)})()

        def close(self):
            pass

    class _Detector:
        def __init__(self):
            self.min_scores = []

        def detect(self, frame, min_score=None):
            self.min_scores.append(min_score)
            return np.array([[20, 10, 80, 40]], dtype=np.float32), np.array([0.9], dtype=np.float32)

    video = tmp_path / "cam_0.mp4"
    video.touch()
    detector = _Detector()
    monkeypatch.setattr(synchronized_timestamps, "read_video_properties", lambda _path: {"fps": 4.0, "frame_count": 5})
    monkeypatch.setattr(object_tracking, "FrameSource", _Source)
    timestamps = SynchronizedTimestamps.from_anchor_frames({0: video}, {0: 0})

    tracks = object_tracking.track_objects({0: video}, detector, timestamps=timestamps, rotation_counts={0: 1}).df

    assert detector.min_scores == [0.1]
    assert tracks.loc[0, ["sync_index", "frame_index", "frame_time", "rotation_count"]].tolist() == [4, 4, 1.0, 1]
    assert tuple(tracks.loc[0, ["box_x1", "box_y1", "box_x2", "box_y2"]]) == pytest.approx(
        _unrotate_box(np.array([20, 10, 80, 40]), 1, 100, 120)
    )


def test_track_objects_leaves_a_length_difference_at_the_end(tmp_path, monkeypatch):
    """A shorter video drops out at its end and shifts no earlier frame."""

    class _Source:
        def __init__(self, wanted_indices):
            self._frames = iter(sorted(wanted_indices))

        @classmethod
        def from_path(cls, *_args, wanted_indices, **_kwargs):
            return cls(wanted_indices)

        def next_frame(self):
            try:
                frame_index = next(self._frames)
            except StopIteration:
                return None
            return type("Raw", (), {"frame_index": frame_index, "frame": np.zeros((20, 30, 3), dtype=np.uint8)})()

        def close(self):
            pass

    class _Detector:
        def detect(self, _frame, min_score=None):
            return np.array([[1, 2, 10, 12]], dtype=np.float32), np.array([0.9], dtype=np.float32)

    videos = {0: tmp_path / "cam_0.mp4", 1: tmp_path / "cam_1.mp4"}
    for video in videos.values():
        video.touch()
    properties = {
        videos[0]: {"fps": 10.0, "frame_count": 5},
        videos[1]: {"fps": 10.0, "frame_count": 3},
    }
    monkeypatch.setattr(synchronized_timestamps, "read_video_properties", properties.__getitem__)
    monkeypatch.setattr(object_tracking, "FrameSource", _Source)
    timestamps = SynchronizedTimestamps.from_anchor_frames(videos, {0: 0, 1: 0})

    tracks = public_track_objects(videos, _Detector(), timestamps=timestamps, frame_step=2, progress=None).df

    by_camera = {cam_id: group.sort_values("frame_index") for cam_id, group in tracks.groupby("cam_id")}
    assert by_camera[0][["sync_index", "frame_index"]].values.tolist() == [[0, 0], [2, 2], [4, 4]]
    assert by_camera[1][["sync_index", "frame_index"]].values.tolist() == [[0, 0], [2, 2]]
    assert by_camera[0]["frame_time"].tolist() == pytest.approx([0.0, 0.2, 0.4])
    assert by_camera[1]["frame_time"].tolist() == pytest.approx([0.0, 0.2])


def test_track_objects_rejects_a_camera_absent_from_timestamps(tmp_path, monkeypatch):
    video_0, video_1 = tmp_path / "cam_0.mp4", tmp_path / "cam_1.mp4"
    video_0.touch()
    video_1.touch()
    monkeypatch.setattr(synchronized_timestamps, "read_video_properties", lambda _path: {"fps": 10.0, "frame_count": 3})
    timestamps = SynchronizedTimestamps.from_anchor_frames({0: video_0}, {0: 0})

    with pytest.raises(ValueError, match="no entry for cameras"):
        public_track_objects({0: video_0, 1: video_1}, object(), timestamps=timestamps, progress=None)


@pytest.mark.parametrize("rotation_count", [1, 2, -1])
def test_keypoint_unrotation_uses_pixel_centres_from_rotated_image(rotation_count: int):
    """A coloured source pixel is returned to its exact source index."""
    source = np.zeros((5, 7, 3), dtype=np.uint8)
    source[2, 4] = (17, 83, 201)
    rotated = object_tracking.apply_rotation(source, rotation_count)
    rotated_y, rotated_x = np.argwhere((rotated == (17, 83, 201)).all(axis=2))[0]

    restored = _unrotate_keypoints(
        np.array([[rotated_x, rotated_y]], dtype=np.float32),
        rotation_count,
        rotated.shape[1],
        rotated.shape[0],
    )

    assert restored[0] == pytest.approx((4, 2))


def test_keypoint_estimation_rejects_box_frame_missing_from_video(tmp_path, monkeypatch):
    class _Source:
        @classmethod
        def from_path(cls, *args, **kwargs):
            return cls()

        def next_frame(self):
            return None

        def close(self):
            pass

    class _KeypointTracker:
        pixel_format = None

    video = tmp_path / "cam_0.mp4"
    video.touch()
    monkeypatch.setattr(object_tracking, "FrameSource", _Source)

    with pytest.raises(ValueError, match="frames unavailable"):
        estimate_keypoints({0: video}, _KeypointTracker(), _box_tracks([_row()]), min_confidence=0)


def test_keypoint_estimation_rejects_changed_source_geometry(tmp_path, monkeypatch):
    class _Source:
        @classmethod
        def from_path(cls, *args, **kwargs):
            return cls()

        def next_frame(self):
            if hasattr(self, "returned"):
                return None
            self.returned = True
            return type("Raw", (), {"frame_index": 7, "frame": np.zeros((119, 100, 3), dtype=np.uint8)})()

        def close(self):
            pass

    class _KeypointTracker:
        pixel_format = None

    video = tmp_path / "cam_0.mp4"
    video.touch()
    monkeypatch.setattr(object_tracking, "FrameSource", _Source)

    with pytest.raises(ValueError, match="image dimensions"):
        estimate_keypoints({0: video}, _KeypointTracker(), _box_tracks([_row()]), min_confidence=0)
