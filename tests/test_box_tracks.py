from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import pytest

from caliscope.core.box_tracks import BOX_TRACK_COLUMNS, BoxTracks
from caliscope.core.track_association import TrackAssociations


def _rows() -> list[dict[str, int | float]]:
    def row(**changes: int | float) -> dict[str, int | float]:
        result: dict[str, int | float] = {
            "sync_index": 0,
            "cam_id": 0,
            "frame_index": 0,
            "frame_time": 0.0,
            "track_id": 1,
            "object_id": -1,
            "box_x1": 10.0,
            "box_y1": 10.0,
            "box_x2": 20.0,
            "box_y2": 20.0,
            "detector_score": 0.9,
            "image_width": 100,
            "image_height": 80,
            "rotation_count": 0,
        }
        result.update(changes)
        return result

    return [
        row(),
        row(sync_index=1, frame_index=1, frame_time=0.1),
        row(track_id=2, object_id=-2, box_x1=20.0, box_y1=20.0, box_x2=50.0, box_y2=50.0, detector_score=0.8),
        row(
            cam_id=4,
            track_id=7,
            object_id=-3,
            box_x1=5.0,
            box_y1=5.0,
            box_x2=15.0,
            box_y2=15.0,
            detector_score=0.7,
            rotation_count=-1,
        ),
    ]


def test_df_is_a_defensive_copy():
    tracks = BoxTracks(pd.DataFrame(_rows()))
    exposed = tracks.df
    exposed.loc[0, "track_id"] = 99

    assert tracks.df.loc[0, "track_id"] == 1


def test_filters_copy_rows_and_edge_exclusions_are_inclusive():
    tracks = BoxTracks(pd.DataFrame(_rows()))

    selected = tracks.filter_to_objects([-1, -3, 99])
    assert set(selected.df["object_id"]) == {-1, -3}
    assert len(selected.df) == 3
    assert tracks.filter_to_objects([]).df.empty

    edge_rows = pd.DataFrame(_rows()[:2])
    edge_rows.loc[0, ["box_x1", "box_y1", "box_x2", "box_y2"]] = [10.0, 11.0, 89.0, 69.0]
    edge_rows.loc[1, ["box_x1", "box_y1", "box_x2", "box_y2"]] = [11.0, 11.0, 89.0, 69.0]
    retained = BoxTracks(edge_rows).exclude_edge_boxes(10.0)
    assert retained.df["frame_index"].tolist() == [1]


def test_csv_round_trip_and_empty_header_only_table(tmp_path):
    path = tmp_path / "box_tracks.csv"
    empty = BoxTracks(pd.DataFrame(columns=BOX_TRACK_COLUMNS))

    empty.to_csv(path)
    reloaded = BoxTracks.from_csv(path)

    assert reloaded.df.empty
    assert reloaded.df.columns.tolist() == list(BOX_TRACK_COLUMNS)


def _track(cam_id: int, sync_indices: Iterable[int]) -> list[dict[str, int | float]]:
    return [
        _rows()[0] | {"sync_index": sync_index, "cam_id": cam_id, "frame_index": sync_index, "object_id": -(cam_id + 1)}
        for sync_index in sync_indices
    ]


def _kept_moments(tracks: BoxTracks, cam_id: int) -> set[int]:
    df = tracks.df
    return {int(moment) for moment in df.loc[df["cam_id"] == cam_id, "sync_index"]}


def test_sample_for_association_keeps_shared_moments_deterministically():
    rows = _track(0, range(20)) + _track(1, range(20)) + _track(2, [50]) + _track(3, [18, 19, 60])
    tracks = BoxTracks(pd.DataFrame(rows))

    sampled = tracks.sample_for_association(frames_per_track=4)

    assert _kept_moments(sampled, 0) == _kept_moments(sampled, 1)
    assert len(_kept_moments(sampled, 0)) >= 4
    assert _kept_moments(sampled, 2) == set()
    assert _kept_moments(sampled, 3) == {18, 19}
    assert sampled.df.equals(tracks.sample_for_association(frames_per_track=4).df)
    assert BoxTracks(pd.DataFrame(_track(0, range(5)))).sample_for_association().df.empty

    for invalid in (0, True):
        with pytest.raises(ValueError, match="frames_per_track"):
            tracks.sample_for_association(frames_per_track=invalid)


def test_sample_for_association_short_track_shares_its_moments_with_long_track():
    tracks = BoxTracks(pd.DataFrame(_track(0, range(100)) + _track(1, range(40, 50))))

    sampled = tracks.sample_for_association(frames_per_track=3)

    short_moments = _kept_moments(sampled, 1)
    assert len(short_moments) >= 3
    assert short_moments <= _kept_moments(sampled, 0)


def test_apply_associations_links_tracks_and_rejects_absent_or_overlapping_ones():
    tracks = BoxTracks(pd.DataFrame(_rows()))
    associations = TrackAssociations.from_mapping({0: {0: [1]}, 1: {4: [7]}})

    linked = tracks.apply_associations(associations).df
    assert sorted(zip(linked["cam_id"], linked["track_id"], linked["object_id"])) == [(0, 1, 0), (0, 1, 0), (4, 7, 1)]

    kept = tracks.apply_associations(associations, keep_unlinked=True).df
    assert len(kept) == len(_rows())
    assert kept.loc[kept["track_id"] == 2, "object_id"].tolist() == [-2]

    with pytest.raises(ValueError, match="absent"):
        tracks.apply_associations(TrackAssociations.from_mapping({0: {9: [1]}}))

    # Tracks 1 and 2 both have a box at camera 0, frame 0.
    with pytest.raises(ValueError, match=r"duplicate \(cam_id, frame_index, object_id\)"):
        tracks.apply_associations(TrackAssociations.from_mapping({0: {0: [1, 2]}}))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("box_x2", 10.0, "positive area"),
        ("image_width", 0, "positive area"),
        ("frame_time", float("inf"), "finite"),
    ],
)
def test_rejects_invalid_box_geometry(column, value, message):
    rows = _rows()[:1]
    rows[0][column] = value

    with pytest.raises(ValueError, match=message):
        BoxTracks(pd.DataFrame(rows))


def test_rejects_duplicate_camera_frame_track_observation():
    rows = _rows()[:1] * 2

    with pytest.raises(ValueError, match="duplicate"):
        BoxTracks(pd.DataFrame(rows))


@pytest.mark.parametrize(
    ("first_changes", "second_changes", "message"),
    [
        ({"object_id": 0}, {"track_id": 2, "object_id": 0}, r"duplicate \(cam_id, frame_index, object_id\)"),
        ({}, {"track_id": 2, "frame_index": 1}, "negative object_id must belong to a single camera track"),
        ({}, {"object_id": -2, "frame_index": 1}, "single object_id"),
    ],
)
def test_rejects_inconsistent_object_ids(first_changes, second_changes, message):
    rows = [_rows()[0] | first_changes, _rows()[0] | second_changes]

    with pytest.raises(ValueError, match=message):
        BoxTracks(pd.DataFrame(rows))


def test_primary_tracks_keep_each_cameras_largest_footprint_as_object_zero():
    primary = BoxTracks(pd.DataFrame(_rows())).primary_tracks().df

    assert sorted(set(zip(primary["cam_id"], primary["track_id"], primary["object_id"]))) == [(0, 2, 0), (4, 7, 0)]

    template = _rows()[0]
    standing = [template | {"sync_index": frame, "frame_index": frame, "frame_time": frame / 10} for frame in range(5)]
    walking = [
        template
        | {"sync_index": frame, "frame_index": frame, "frame_time": frame / 10, "track_id": 2, "object_id": -2}
        | {"box_x1": 10.0 + 10 * frame, "box_x2": 20.0 + 10 * frame}
        for frame in range(2)
    ]
    crossing = BoxTracks(pd.DataFrame(standing + walking)).primary_tracks().df
    assert set(crossing["track_id"]) == {2}
