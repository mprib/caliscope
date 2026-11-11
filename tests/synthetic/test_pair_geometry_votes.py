"""Stages 1 to 3 of the unposed bootstrap on synthetic tracked scenes: propose, vote, close over triplets.

The stages see only unlinked points and the intrinsics of the cameras. Assertions
check which track pairs the chosen geometries match against the scene's truth.
Geometry errors against the true rig are printed (run with -s) and asserted only
once ROTATION_TOL_DEG and TRANSLATION_TOL_DEG are set from the 20-seed measurement
of the spec's section 7.3; until then they are provisional and not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.pair_geometry_votes import (
    STATUS_ACCEPTED,
    _ClosureResult,
    _close_triplets,
    _index_pixel_tracks,
    _propose_pair_geometries,
    _vote,
)
from caliscope.core.box_tracks import BoxTracks
from caliscope.core.point_data import ImagePoints
from caliscope.core.track_association import _moments_by_track
from caliscope.synthetic import SyntheticScene
from tests.synthetic.tracked_scene import (
    TrackKey,
    build_scene,
    direction_error_deg,
    relative_pose,
    ring_cameras,
    rotation_error_deg,
    sample_points,
    to_tracked,
    walking,
)

N_FRAMES = 40
FRAMES_PER_TRACK = 20
SEEDS = range(5)  # the spec's measurement uses range(20)

# Provisional, pending the 20-seed measurement. None means the assertion is skipped.
ROTATION_TOL_DEG: float | None = None
TRANSLATION_TOL_DEG: float | None = None


def _run_stages(points: ImagePoints, boxes: BoxTracks, cameras: CameraArray) -> _ClosureResult:
    sampled = sample_points(points, boxes, FRAMES_PER_TRACK)
    tracks = _index_pixel_tracks(sampled, cameras)
    voted = _vote(_propose_pair_geometries(tracks, cameras), cameras)
    return _close_triplets(voted, _moments_by_track(boxes.df), sorted({cam_id for cam_id, _ in tracks}))


def _matches_true_objects(result: _ClosureResult, truth: dict[TrackKey, int]) -> dict[tuple[int, int], bool]:
    return {
        pair: all(truth[(pair[0], ta)] == truth[(pair[1], tb)] for ta, tb in decision.chosen.matching)
        for pair, decision in result.decisions.items()
    }


def _geometry_errors(result: _ClosureResult, scene: SyntheticScene) -> dict[tuple[int, int], tuple[float, float]]:
    cameras = scene.camera_array.cameras
    errors: dict[tuple[int, int], tuple[float, float]] = {}
    for pair, decision in result.decisions.items():
        rotation, translation = relative_pose(cameras[pair[0]], cameras[pair[1]])
        proposal = decision.chosen.proposal
        errors[pair] = (
            rotation_error_deg(proposal.rotation, rotation),
            direction_error_deg(proposal.translation, translation),
        )
    return errors


def _check_geometry(errors: dict[tuple[int, int], tuple[float, float]], label: str) -> None:
    print(f"{label} worst errors (rotation deg, translation direction deg): {errors}")
    for pair, (rotation_deg, translation_deg) in errors.items():
        if ROTATION_TOL_DEG is not None:
            assert rotation_deg <= ROTATION_TOL_DEG, f"pair {pair}: rotation error {rotation_deg:.3f} deg"
        if TRANSLATION_TOL_DEG is not None:
            assert translation_deg <= TRANSLATION_TOL_DEG, f"pair {pair}: translation error {translation_deg:.3f} deg"


@pytest.mark.parametrize("seed", SEEDS)
def test_two_walkers_four_cameras_choose_true_geometries_and_pairings(seed: int):
    cameras = ring_cameras(n=4)
    scene = build_scene(
        cameras,
        [walking((-1.2, -0.9), (1.2, -0.9), N_FRAMES), walking((1.0, 0.9), (-1.0, 0.9), N_FRAMES)],
    )
    points, boxes, truth = to_tracked(scene, seed=seed)

    result = _run_stages(points, boxes, cameras)

    assert sorted(result.decisions) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert result.placed_cameras == (0, 1, 2, 3)
    assert all(decision.accepted for decision in result.decisions.values())
    assert all(_matches_true_objects(result, truth).values())
    assert all(decision.chosen.n_voters == 2 for decision in result.decisions.values())
    assert all(decision.identity_contradictions == 0 for decision in result.decisions.values())
    _check_geometry(_geometry_errors(result, scene), f"seed {seed}")


def test_swapped_pairing_of_people_in_step_is_not_chosen():
    # Cameras 0 and 1 sit at (5, 0) and (0, 5), so their baseline runs along u.
    # The two people walk together, offset 0.9 m along u, so a swapped match
    # between cameras 0 and 1 fits an epipolar geometry of its own.
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
    scene = build_scene(cameras, trajectories)
    points, boxes, truth = to_tracked(scene)

    result = _run_stages(points, boxes, cameras)

    swapped = [
        voted
        for voted in result.kept[(0, 1)]
        if truth[(0, voted.proposal.track_a)] != truth[(1, voted.proposal.track_b)]
    ]
    assert swapped, "no swapped proposal was kept for cameras (0, 1)"
    chosen = result.decisions[(0, 1)].chosen
    assert truth[(0, chosen.proposal.track_a)] == truth[(1, chosen.proposal.track_b)]
    assert all(_matches_true_objects(result, truth).values())
    _check_geometry(_geometry_errors(result, scene), "in-step walkers")


def test_two_cameras_skip_closure_and_the_vote_picks_the_true_pairing():
    cameras = ring_cameras(n=4, drop=(2, 3))
    scene = build_scene(
        cameras,
        [walking((-1.2, -0.9), (1.2, -0.9), N_FRAMES), walking((1.0, 0.9), (-1.0, 0.9), N_FRAMES)],
    )
    points, boxes, truth = to_tracked(scene)

    result = _run_stages(points, boxes, cameras)

    decision = result.decisions[(0, 1)]
    assert decision.status == STATUS_ACCEPTED
    assert decision.accepted
    assert result.placed_cameras == (0, 1)
    assert result.unplaced_cameras == ()
    assert decision.chosen.n_voters == 2
    assert _matches_true_objects(result, truth) == {(0, 1): True}
    _check_geometry(_geometry_errors(result, scene), "two cameras")
