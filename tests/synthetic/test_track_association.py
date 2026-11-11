"""associate_tracks on synthetic multi-object scenes with known extrinsics.

Each scene is converted to camera-local tracks (tests/synthetic/tracked_scene.py),
sampled the way a user samples them, and linked with the default reprojection
gate. Assertions check outcomes against the scene's ground truth: which true
object each linked object shows, which cameras it spans, and why tracks stay
unresolved. Every test also checks exclusivity: a track appears once, and the
links apply to the boxes without breaking BoxTracks validation.
"""

from __future__ import annotations

import numpy as np

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.box_tracks import BoxTracks
from caliscope.core.point_data import ImagePoints
from caliscope.core.track_association import (
    CONFLICTING_TRACK,
    TrackAssociations,
    _index_tracks,
    _pairwise_seeds,
    _Rig,
    associate_tracks,
)
from caliscope.synthetic import SyntheticScene
from tests.synthetic.tracked_scene import (
    assert_exclusive,
    build_scene,
    cameras_by_object,
    footprint_by_true_object,
    identities_by_object,
    median_box_area,
    object_of_track,
    ring_cameras,
    sample_points,
    standing,
    to_tracked,
    turning_in_place,
    walking,
)

N_FRAMES = 30
SUBJECT = 0
OTHER = 1


def _associate(points: ImagePoints, boxes: BoxTracks, cameras: CameraArray) -> TrackAssociations:
    associations = associate_tracks(sample_points(points, boxes), cameras, boxes)
    assert_exclusive(associations, boxes)
    return associations


def _object_showing(identities: dict[int, set[int]], true_object: int) -> int:
    matches = [object_id for object_id, shown in identities.items() if shown == {true_object}]
    assert len(matches) == 1, f"expected one object showing true object {true_object}, got {identities}"
    return matches[0]


def _two_walkers(cameras: CameraArray) -> SyntheticScene:
    """A subject and a second person walking in opposite directions on parallel lines 1.8 m apart."""
    return build_scene(
        cameras,
        [walking((-1.2, -0.9), (1.2, -0.9), N_FRAMES), walking((1.0, 0.9), (-1.0, 0.9), N_FRAMES)],
    )


def test_two_coherent_objects_link_on_gapped_cam_ids_and_arbitrary_sync_indices():
    cameras = ring_cameras(n=8, drop=(0, 3, 4, 6))
    assert set(cameras.cameras) == {1, 2, 5, 7}
    scene = build_scene(
        cameras,
        [walking((-1.2, -0.9), (1.2, -0.9), N_FRAMES), walking((1.0, 0.9), (-1.0, 0.9), N_FRAMES)],
        scales=[0.8, 1.0],
    )
    points, boxes, truth = to_tracked(scene, sync_offset=100, sync_stride=3)

    associations = _associate(points, boxes, cameras)

    identities = identities_by_object(associations, truth)
    assert sorted(identities) == [0, 1]
    assert sorted(identities.values(), key=min) == [{0}, {1}]
    assert cameras_by_object(associations) == {0: {1, 2, 5, 7}, 1: {1, 2, 5, 7}}
    assert associations.unresolved.empty

    footprints = footprint_by_true_object(boxes, truth)
    assert identities[0] == {max(footprints, key=footprints.__getitem__)}


def test_dominant_bystander_does_not_mix_identities_and_ranking_follows_footprint():
    cameras = ring_cameras(n=4)
    bystander = OTHER
    scene = build_scene(
        cameras,
        [walking((-1.0, -1.5), (1.0, -1.5), N_FRAMES), turning_in_place((1.5, 1.5), N_FRAMES)],
    )
    points, boxes, truth = to_tracked(scene, hidden={(2, bystander), (3, bystander)})
    for cam_id in (0, 1):
        assert median_box_area(boxes, truth, cam_id, bystander) > median_box_area(boxes, truth, cam_id, SUBJECT)

    associations = _associate(points, boxes, cameras)

    identities = identities_by_object(associations, truth)
    assert all(len(shown) == 1 for shown in identities.values()), identities
    subject = _object_showing(identities, SUBJECT)
    linked_bystander = _object_showing(identities, bystander)
    assert cameras_by_object(associations)[subject] == {0, 1, 2, 3}
    assert cameras_by_object(associations)[linked_bystander] == {0, 1}

    footprints = footprint_by_true_object(boxes, truth)
    expected_order = sorted(footprints, key=lambda true_object: -footprints[true_object])
    assert [next(iter(identities[object_id])) for object_id in sorted(identities)] == expected_order


def test_split_track_fragments_join_the_same_object():
    cameras = ring_cameras(n=4)
    points, boxes, truth = to_tracked(_two_walkers(cameras), splits={(2, SUBJECT): 14})
    fragments = sorted(key for key, true_object in truth.items() if key[0] == 2 and true_object == SUBJECT)
    assert len(fragments) == 2

    associations = _associate(points, boxes, cameras)

    identities = identities_by_object(associations, truth)
    subject = _object_showing(identities, SUBJECT)
    assert [object_of_track(associations, key) for key in fragments] == [subject, subject]
    assert cameras_by_object(associations)[subject] == {0, 1, 2, 3}


def test_overlapping_duplicate_track_leaves_both_unresolved_as_a_conflict():
    cameras = ring_cameras(n=4)
    points, boxes, truth = to_tracked(_two_walkers(cameras), duplicates={(2, SUBJECT): (0, N_FRAMES)})
    original, duplicate = sorted(key for key, true_object in truth.items() if key[0] == 2 and true_object == SUBJECT)

    associations = _associate(points, boxes, cameras)

    unresolved = associations.unresolved
    reasons = {
        (int(cam_id), int(track_id)): reason
        for cam_id, track_id, reason in zip(unresolved["cam_id"], unresolved["track_id"], unresolved["reason"])
    }
    assert reasons[original] == CONFLICTING_TRACK.format(other=duplicate[1])
    assert reasons[duplicate] == CONFLICTING_TRACK.format(other=original[1])

    identities = identities_by_object(associations, truth)
    subject = _object_showing(identities, SUBJECT)
    assert cameras_by_object(associations)[subject] == {0, 1, 3}


def test_people_side_by_side_along_one_epipolar_plane_are_resolved_by_further_views():
    # Cameras 0 and 1 sit at (5, 0) and (0, 5), so their baseline runs along u.
    # The two people walk together, offset 0.9 m along u, so every keypoint of
    # one lies in the same epipolar plane as the matching keypoint of the other:
    # in the pair (0, 1), and likewise (2, 3), a swapped match fits exactly.
    cameras = ring_cameras(n=4)
    along_baseline = np.array([-1.0, 1.0]) / np.sqrt(2)
    walk_direction = np.array([1.0, 1.0]) / np.sqrt(2)
    trajectories = [
        walking(
            (-1.2 * walk_direction + side * along_baseline).tolist(),
            (1.2 * walk_direction + side * along_baseline).tolist(),
            N_FRAMES,
        )
        for side in (-0.45, 0.45)
    ]
    points, boxes, truth = to_tracked(build_scene(cameras, trajectories))

    associations = _associate(points, boxes, cameras)

    identities = identities_by_object(associations, truth)
    assert sorted(identities.values(), key=min) == [{0}, {1}]
    assert all(linked_cameras == {0, 1, 2, 3} for linked_cameras in cameras_by_object(associations).values())


def test_nearly_static_subject_links_in_every_camera():
    cameras = ring_cameras(n=4)
    points, boxes, truth = to_tracked(build_scene(cameras, [standing((0.3, -0.2), N_FRAMES)]))

    associations = _associate(points, boxes, cameras)

    assert identities_by_object(associations, truth) == {0: {SUBJECT}}
    assert cameras_by_object(associations) == {0: {0, 1, 2, 3}}


def test_best_ranked_seed_of_the_static_scene_is_not_a_facing_pair():
    cameras = ring_cameras(n=4)
    points, boxes, _ = to_tracked(build_scene(cameras, [standing((0.3, -0.2), N_FRAMES)]))
    df = sample_points(points, boxes).df
    posed = cameras.posed_cameras
    rig = _Rig.from_cameras({cam_id: posed[cam_id] for cam_id in sorted(set(df["cam_id"].tolist()))})
    seeds = _pairwise_seeds(_index_tracks(df, posed, rig), rig, 8.0)

    first = min(seeds, key=lambda pair: (seeds[pair].rank, pair))

    assert seeds[first].n_conditioned_moments == seeds[first].n_moments > 0, f"first seed {first} is a facing pair"


def test_bad_keypoint_still_links_and_has_the_lowest_inlier_rate():
    bad_cam, bad_keypoint = 2, 9
    cameras = ring_cameras(n=4)
    points, boxes, truth = to_tracked(build_scene(cameras, [walking((-1.2, -0.9), (1.2, -0.9), N_FRAMES)]))
    df = points.df
    df.loc[(df["cam_id"] == bad_cam) & (df["keypoint_id"] == bad_keypoint), "img_loc_x"] += 30.0
    points = ImagePoints(df)

    associations = _associate(points, boxes, cameras)

    assert identities_by_object(associations, truth) == {0: {SUBJECT}}
    assert cameras_by_object(associations) == {0: {0, 1, 2, 3}}
    rates = associations.keypoint_inlier_rates
    rates = rates[rates["object_id"] == 0].set_index("keypoint_id")["inlier_rate"]
    assert rates[bad_keypoint] < rates.drop(bad_keypoint).min()


def test_two_camera_rig_links_subject_and_bystander():
    cameras = ring_cameras(n=4, drop=(2, 3))
    scene = build_scene(
        cameras,
        [walking((-1.2, -0.8), (1.2, -0.8), N_FRAMES), turning_in_place((0.8, 1.2), N_FRAMES)],
    )
    points, boxes, truth = to_tracked(scene)

    associations = _associate(points, boxes, cameras)

    identities = identities_by_object(associations, truth)
    assert sorted(identities.values(), key=min) == [{0}, {1}]
    assert cameras_by_object(associations) == {0: {0, 1}, 1: {0, 1}}
