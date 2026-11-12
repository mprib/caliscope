"""TrackAssociations persistence and associate_tracks input checks (no geometry)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.box_tracks import BoxTracks
from caliscope.core.point_data import ImagePoints
from caliscope.core.track_association import NO_EXTRINSICS, TrackAssociations, _cluster_tracks, associate_tracks
from caliscope.exceptions import CalibrationError
from caliscope.synthetic import CameraSynthesizer


def test_hand_built_associations_round_trip_with_statistics_omitted(tmp_path):
    associations = TrackAssociations.from_mapping({1: {5: [2]}, 0: {1: [4, 3]}})
    path = tmp_path / "associations.toml"

    associations.to_toml(path)
    reloaded = TrackAssociations.from_toml(path)

    assert "n_moments" not in path.read_text()
    pd.testing.assert_frame_equal(reloaded.linked, associations.linked)
    assert reloaded.linked[["object_id", "cam_id", "track_id"]].values.tolist() == [[0, 1, 3], [0, 1, 4], [1, 5, 2]]
    assert reloaded.linked["n_moments"].isna().all()
    assert reloaded.unresolved.empty
    assert reloaded.keypoint_inlier_rates.empty


def test_associations_with_statistics_round_trip(tmp_path):
    linked = pd.DataFrame(
        [
            {
                "object_id": 0,
                "cam_id": 1,
                "track_id": 4,
                "n_moments": 22,
                "n_observations": 431,
                "inlier_fraction": 0.91,
                "median_reprojection_px": 2.3,
            },
            {"object_id": 1, "cam_id": 5, "track_id": 2},
        ]
    )
    unresolved = pd.DataFrame([{"cam_id": 2, "track_id": 9, "reason": "no geometric support"}])
    rates = pd.DataFrame([{"object_id": 0, "keypoint_id": 9, "n_observations": 120, "inlier_rate": 0.42}])
    associations = TrackAssociations(linked, unresolved, rates)
    path = tmp_path / "associations.toml"

    associations.to_toml(path)
    reloaded = TrackAssociations.from_toml(path)

    pd.testing.assert_frame_equal(reloaded.linked, associations.linked)
    pd.testing.assert_frame_equal(reloaded.unresolved, associations.unresolved)
    pd.testing.assert_frame_equal(reloaded.keypoint_inlier_rates, associations.keypoint_inlier_rates)
    assert reloaded.linked["n_moments"].tolist()[0] == 22
    assert pd.isna(reloaded.linked["n_moments"].tolist()[1])


@pytest.mark.parametrize("mapping", [{0: {1: [4]}, 1: {1: [4]}}, {-1: {1: [4]}}])
def test_from_mapping_rejects_a_shared_track_or_negative_object(mapping):
    with pytest.raises(ValueError):
        TrackAssociations.from_mapping(mapping)


def _tiny_inputs() -> tuple[pd.DataFrame, CameraArray, BoxTracks]:
    """Three posed cameras, one unlinked track each, three moments of three keypoints."""
    cameras = CameraSynthesizer().add_ring(n=3, radius=3.0, height=1.0).build()
    rng = np.random.default_rng(0)
    point_rows: list[dict[str, int | float]] = []
    box_rows: list[dict[str, int | float]] = []
    for object_id, cam_id in ((-1, 0), (-2, 1), (-3, 2)):
        for sync_index in range(3):
            for keypoint_id in range(3):
                point_rows.append(
                    {
                        "sync_index": sync_index,
                        "cam_id": cam_id,
                        "object_id": object_id,
                        "track_id": 1,
                        "keypoint_id": keypoint_id,
                        "img_loc_x": float(rng.uniform(800, 1100)),
                        "img_loc_y": float(rng.uniform(400, 700)),
                    }
                )
            box_rows.append(
                {
                    "sync_index": sync_index,
                    "cam_id": cam_id,
                    "frame_index": sync_index,
                    "frame_time": sync_index / 30,
                    "track_id": 1,
                    "object_id": object_id,
                    "box_x1": 700.0,
                    "box_y1": 300.0,
                    "box_x2": 1200.0,
                    "box_y2": 800.0,
                    "detector_score": 0.9,
                    "image_width": 1920,
                    "image_height": 1080,
                    "rotation_count": 0,
                }
            )
    return pd.DataFrame(point_rows), cameras, BoxTracks(pd.DataFrame(box_rows))


def _remove_extrinsics(cameras: CameraArray, cam_ids: list[int]) -> None:
    for cam_id in cam_ids:
        cameras[cam_id].rotation = None
        cameras[cam_id].translation = None


def test_associate_tracks_rejects_points_without_track_id_or_with_linked_rows():
    points, cameras, boxes = _tiny_inputs()

    with pytest.raises(ValueError, match="track_id"):
        associate_tracks(ImagePoints(points.drop(columns=["track_id"])), cameras, boxes)

    linked = points.copy()
    linked.loc[0, "object_id"] = 0
    with pytest.raises(ValueError, match="unlinked"):
        associate_tracks(ImagePoints(linked), cameras, boxes)


def test_associate_tracks_needs_two_cameras_with_extrinsics():
    points, cameras, boxes = _tiny_inputs()
    _remove_extrinsics(cameras, [1, 2])

    with pytest.raises(CalibrationError, match="bootstrap_extrinsics"):
        associate_tracks(ImagePoints(points), cameras, boxes)


def test_tracks_in_a_camera_without_extrinsics_are_unresolved():
    points, cameras, boxes = _tiny_inputs()
    _remove_extrinsics(cameras, [2])

    associations = associate_tracks(ImagePoints(points), cameras, boxes)

    unresolved = associations.unresolved
    assert unresolved.loc[unresolved["cam_id"] == 2, "reason"].tolist() == [NO_EXTRINSICS]
    tracks = list(zip(associations.linked["cam_id"], associations.linked["track_id"])) + list(
        zip(unresolved["cam_id"], unresolved["track_id"])
    )
    assert sorted(tracks) == [(0, 1), (1, 1), (2, 1)]


def _weights(n_tracks: int, pairs: dict[tuple[int, int], float]) -> np.ndarray:
    weights = np.zeros((n_tracks, n_tracks))
    for (a, b), weight in pairs.items():
        weights[a, b] = weights[b, a] = weight
    return weights


def _cannot_link(n_tracks: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    blocked = np.zeros((n_tracks, n_tracks), dtype=bool)
    for a, b in pairs:
        blocked[a, b] = blocked[b, a] = True
    return blocked


def test_clustering_keeps_a_track_whose_one_bad_pair_is_outvoted():
    # Four cameras, one track each. Tracks 0 and 2 disagree (cameras facing each other), every other pair agrees.
    agree = {(a, b): 0.8 for a in range(4) for b in range(a + 1, 4)}
    weights = _weights(4, agree | {(0, 2): -0.5})

    clusters, rivals = _cluster_tracks(weights, _cannot_link(4, []))

    assert clusters == [[0, 1, 2, 3]]
    assert rivals == {}


def test_clustering_settles_an_ambiguous_pair_from_the_other_cameras():
    # Two people (tracks 0-2 and 3-5) in three cameras. In the first two cameras a swapped match fits as well as
    # the true one; the third camera tells them apart.
    person_a, person_b = [0, 1, 2], [3, 4, 5]
    pairs = {(a, b): -1.0 for a in person_a for b in person_b}
    pairs |= {(0, 1): 0.9, (0, 2): 0.9, (1, 2): 0.9, (3, 4): 0.9, (3, 5): 0.9, (4, 5): 0.9}
    pairs |= {(0, 4): 0.9, (1, 3): 0.9}
    cannot_link = _cannot_link(6, [(0, 3), (1, 4), (2, 5)])

    clusters, rivals = _cluster_tracks(_weights(6, pairs), cannot_link)

    assert clusters == [person_a, person_b]
    assert rivals == {}


def test_clustering_leaves_out_two_overlapping_tracks_that_fit_equally():
    # Tracks 2 and 3 overlap in one camera and both agree with tracks 0 and 1.
    pairs = {(0, 1): 0.9, (0, 2): 0.8, (1, 2): 0.8, (0, 3): 0.75, (1, 3): 0.75}

    clusters, rivals = _cluster_tracks(_weights(4, pairs), _cannot_link(4, [(2, 3)]))

    assert clusters == [[0, 1], [2], [3]]
    assert rivals == {2: 3, 3: 2}
