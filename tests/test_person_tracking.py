"""Focused checks for box-first person tracking helpers and boundaries."""

import numpy as np
import pandas as pd
import pytest

from caliscope.api import track_people as public_track_people
from caliscope.core.box_tracks import BoxTracks
from caliscope.recording import person_tracking
from caliscope.recording.person_tracking import (
    _empty_image_points,
    _rotate_box,
    _unrotate_keypoints,
    _unrotate_box,
    _validate_selected_tracks,
    estimate_keypoints,
)


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


def _row(*, cam_id: int = 0, track_id: int = 1) -> dict:
    return {
        "sync_index": 3,
        "cam_id": cam_id,
        "frame_index": 7,
        "frame_time": 1.2,
        "track_id": track_id,
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


def test_multiple_tracks_are_rejected_before_pose_confidence_filtering():
    first = _row(track_id=1)
    second = _row(track_id=2)
    second["frame_index"] = 8
    tracks = _box_tracks([first, second])

    with pytest.raises(ValueError, match="at most one track per camera"):
        _validate_selected_tracks(tracks.df)


def test_keypoint_estimation_reports_table_camera_absent_from_videos():
    tracks = _box_tracks([_row(cam_id=5)])

    with pytest.raises(ValueError, match="absent from videos"):
        estimate_keypoints({}, object(), tracks, min_confidence=0.5)


@pytest.mark.parametrize("min_confidence", [-0.01, np.nan, np.inf])
def test_keypoint_estimation_requires_a_finite_nonnegative_cutoff(min_confidence: float):
    tracks = _box_tracks([])

    with pytest.raises(ValueError, match="finite and non-negative"):
        estimate_keypoints({}, object(), tracks, min_confidence=min_confidence)


def test_track_people_records_source_coordinates_and_frame_metadata(tmp_path, monkeypatch):
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

        def people(self, frame, min_score=None):
            self.min_scores.append(min_score)
            return np.array([[20, 10, 80, 40]], dtype=np.float32), np.array([0.9], dtype=np.float32)

    video = tmp_path / "cam_0.mp4"
    video.touch()
    detector = _Detector()
    monkeypatch.setattr(person_tracking, "read_video_properties", lambda _path: {"fps": 4.0, "frame_count": 5})
    monkeypatch.setattr(person_tracking, "FrameSource", _Source)

    tracks = person_tracking.track_people({0: video}, detector, rotation_counts={0: 1}).df

    assert detector.min_scores == [0.1]
    assert tracks.loc[0, ["sync_index", "frame_index", "frame_time", "rotation_count"]].tolist() == [4, 4, 1.0, 1]
    assert tuple(tracks.loc[0, ["box_x1", "box_y1", "box_x2", "box_y2"]]) == pytest.approx(
        _unrotate_box(np.array([20, 10, 80, 40]), 1, 100, 120)
    )


def test_track_people_samples_each_video_at_source_frame_indices(tmp_path, monkeypatch):
    """Each video keeps its own FPS without duration normalization."""

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
        def people(self, _frame, min_score=None):
            return np.array([[1, 2, 10, 12]], dtype=np.float32), np.array([0.9], dtype=np.float32)

    videos = {0: tmp_path / "cam_0.mp4", 1: tmp_path / "cam_1.mp4"}
    for video in videos.values():
        video.touch()
    properties = {
        videos[0]: {"fps": 10.0, "frame_count": 5},
        videos[1]: {"fps": 20.0, "frame_count": 5},
    }
    monkeypatch.setattr(person_tracking, "read_video_properties", properties.__getitem__)
    monkeypatch.setattr(person_tracking, "FrameSource", _Source)

    tracks = public_track_people(videos, _Detector(), frame_step=2, progress=None).df

    by_camera = {cam_id: group.sort_values("frame_index") for cam_id, group in tracks.groupby("cam_id")}
    assert by_camera[0][["sync_index", "frame_index"]].values.tolist() == [[0, 0], [2, 2], [4, 4]]
    assert by_camera[1][["sync_index", "frame_index"]].values.tolist() == [[0, 0], [2, 2], [4, 4]]
    assert by_camera[0]["frame_time"].tolist() == pytest.approx([0.0, 0.2, 0.4])
    assert by_camera[1]["frame_time"].tolist() == pytest.approx([0.0, 0.1, 0.2])


def test_track_people_truncates_videos_to_the_shortest(tmp_path, monkeypatch):
    """Frames past the shortest video's end are never requested from any camera."""
    requested = {}

    class _Source:
        @classmethod
        def from_path(cls, path, *_args, wanted_indices, **_kwargs):
            requested[path.name] = sorted(wanted_indices)
            return cls()

        def next_frame(self):
            return None

        def close(self):
            pass

    videos = {0: tmp_path / "cam_0.mp4", 1: tmp_path / "cam_1.mp4"}
    for video in videos.values():
        video.touch()
    properties = {
        videos[0]: {"fps": 10.0, "frame_count": 5},
        videos[1]: {"fps": 10.0, "frame_count": 3},
    }
    monkeypatch.setattr(person_tracking, "read_video_properties", properties.__getitem__)
    monkeypatch.setattr(person_tracking, "FrameSource", _Source)

    public_track_people(videos, object(), progress=None)

    assert requested == {"cam_0.mp4": [0, 1, 2], "cam_1.mp4": [0, 1, 2]}


@pytest.mark.parametrize("rotation_count", [1, 2, -1])
def test_keypoint_unrotation_uses_pixel_centres_from_rotated_image(rotation_count: int):
    """A coloured source pixel is returned to its exact source index."""
    source = np.zeros((5, 7, 3), dtype=np.uint8)
    source[2, 4] = (17, 83, 201)
    rotated = person_tracking.apply_rotation(source, rotation_count)
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

    class _PoseTracker:
        pixel_format = None

    video = tmp_path / "cam_0.mp4"
    video.touch()
    monkeypatch.setattr(person_tracking, "FrameSource", _Source)

    with pytest.raises(ValueError, match="frames unavailable"):
        estimate_keypoints({0: video}, _PoseTracker(), _box_tracks([_row()]), min_confidence=0)


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

    class _PoseTracker:
        pixel_format = None

    video = tmp_path / "cam_0.mp4"
    video.touch()
    monkeypatch.setattr(person_tracking, "FrameSource", _Source)

    with pytest.raises(ValueError, match="image dimensions"):
        estimate_keypoints({0: video}, _PoseTracker(), _box_tracks([_row()]), min_confidence=0)
