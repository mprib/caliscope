from __future__ import annotations

import pandas as pd
import pytest

from caliscope.core.box_tracks import BOX_TRACK_COLUMNS, BoxTracks


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
