"""associate_tracks on synthetic multi-object scenes with known extrinsics.

Each scene is converted to camera-local tracks (tests/synthetic/tracked_scene.py)
and sampled the way a user samples them. The embeddings are made up: boxes of
one true object share a vector. So these tests check the contract of
associate_tracks, not how well it tells people apart. That evidence is the
measurement on real footage.

Assertions check outcomes against the scene's ground truth: which true object
each linked object shows, which cameras it spans, and why tracks stay
unresolved. Every test also checks exclusivity: a track appears once, and the
links apply to the boxes without breaking BoxTracks validation.
"""

from __future__ import annotations

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.box_tracks import BoxTracks
from caliscope.core.point_data import ImagePoints
from caliscope.core.track_association import (
    CONFLICTING_TRACK,
    TrackAssociations,
    associate_tracks,
)
from caliscope.synthetic import SyntheticScene
from tests.synthetic.tracked_scene import (
    assert_exclusive,
    build_scene,
    cameras_by_object,
    embeddings_by_true_object,
    footprint_by_true_object,
    identities_by_object,
    median_box_area,
    object_of_track,
    ring_cameras,
    sample_points,
    to_tracked,
    turning_in_place,
    walking,
)

N_FRAMES = 30
SUBJECT = 0
OTHER = 1


def _associate(
    points: ImagePoints, boxes: BoxTracks, cameras: CameraArray, truth: dict[tuple[int, int], int]
) -> TrackAssociations:
    sampled = sample_points(points, boxes)
    associations = associate_tracks(sampled, cameras, boxes, embeddings_by_true_object(sampled, truth))
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

    associations = _associate(points, boxes, cameras, truth)

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

    associations = _associate(points, boxes, cameras, truth)

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

    associations = _associate(points, boxes, cameras, truth)

    identities = identities_by_object(associations, truth)
    subject = _object_showing(identities, SUBJECT)
    assert [object_of_track(associations, key) for key in fragments] == [subject, subject]
    assert cameras_by_object(associations)[subject] == {0, 1, 2, 3}


def test_overlapping_duplicate_track_leaves_both_unresolved_as_a_conflict():
    cameras = ring_cameras(n=4)
    points, boxes, truth = to_tracked(_two_walkers(cameras), duplicates={(2, SUBJECT): (0, N_FRAMES)})
    original, duplicate = sorted(key for key, true_object in truth.items() if key[0] == 2 and true_object == SUBJECT)

    associations = _associate(points, boxes, cameras, truth)

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


def test_two_camera_rig_links_subject_and_bystander():
    cameras = ring_cameras(n=4, drop=(2, 3))
    scene = build_scene(
        cameras,
        [walking((-1.2, -0.8), (1.2, -0.8), N_FRAMES), turning_in_place((0.8, 1.2), N_FRAMES)],
    )
    points, boxes, truth = to_tracked(scene)

    associations = _associate(points, boxes, cameras, truth)

    identities = identities_by_object(associations, truth)
    assert sorted(identities.values(), key=min) == [{0}, {1}]
    assert cameras_by_object(associations) == {0: {0, 1}, 1: {0, 1}}
