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
        row(track_id=2, box_x1=20.0, box_y1=20.0, box_x2=50.0, box_y2=50.0, detector_score=0.8),
        row(
            cam_id=4,
            track_id=7,
            box_x1=5.0,
            box_y1=5.0,
            box_x2=15.0,
            box_y2=15.0,
            detector_score=0.7,
            rotation_count=-1,
        ),
    ]


def test_primary_track_ids_uses_covered_area_and_stable_ties():
    tracks = BoxTracks(pd.DataFrame(_rows()))

    assert tracks.primary_track_ids() == {0: 2, 4: 7}

    tied = pd.DataFrame(_rows()[:2])
    tied.loc[1, "track_id"] = 3
    assert BoxTracks(tied).primary_track_ids() == {0: 1}


def test_primary_track_ids_counts_repeated_coverage_once():
    template = _rows()[0]
    standing = [
        template | {"sync_index": frame, "frame_index": frame, "frame_time": frame / 10, "track_id": 1}
        for frame in range(5)
    ]
    walking = [
        template
        | {
            "track_id": 2,
            "sync_index": frame,
            "frame_index": frame,
            "frame_time": frame / 10,
            "box_x1": 10.0 + 12 * frame,
            "box_x2": 19.0 + 12 * frame,
        }
        for frame in range(2)
    ]

    assert BoxTracks(pd.DataFrame(standing + walking)).primary_track_ids() == {0: 2}


def test_df_is_a_defensive_copy():
    tracks = BoxTracks(pd.DataFrame(_rows()))
    exposed = tracks.df
    exposed.loc[0, "track_id"] = 99

    assert tracks.df.loc[0, "track_id"] == 1


def test_filters_copy_rows_and_edge_exclusions_are_inclusive():
    tracks = BoxTracks(pd.DataFrame(_rows()))

    selected = tracks.filter_to_tracks({0: 1})
    assert set(selected.df["track_id"]) == {1}
    assert set(selected.df["cam_id"]) == {0}
    with pytest.raises(ValueError, match="absent"):
        tracks.filter_to_tracks({4: 1})
    with pytest.raises(ValueError, match="absent"):
        tracks.filter_to_tracks({0: 1.5})

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
