"""Canaries for the per-moment multi-way matching."""

import numpy as np

from caliscope.core.moment_matching import NEUTRAL_AFFINITY, geometric_affinity, match_moment


def _affinity(identity: np.ndarray, same: float = 0.85, different: float = 0.15) -> np.ndarray:
    return np.where(identity[:, None] == identity[None, :], same, different).astype(float)


def test_people_are_grouped_across_three_views_and_a_lone_box_is_left_out():
    identity = np.array([0, 1, 2, 0, 1, 2, 0, 1, 9])
    views = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    result = match_moment(_affinity(identity), views)

    assert set(result.groups) == {(0, 3, 6), (1, 4, 7), (2, 5)}
    assert result.inconsistent == ()


def test_one_strong_wrong_pair_is_overruled_by_the_other_views():
    identity = np.array([0, 1, 0, 1, 0, 1])
    views = np.array([0, 0, 1, 1, 2, 2])
    affinity = _affinity(identity)
    affinity[0, 3] = affinity[3, 0] = 0.9  # box 0 also looks like the other person in view 1

    result = match_moment(affinity, views)

    assert set(result.groups) == {(0, 2, 4), (1, 3, 5)}


def test_two_views_match_only_above_the_neutral_affinity():
    views = np.array([0, 1])
    above = np.array([[0.0, NEUTRAL_AFFINITY + 0.1], [NEUTRAL_AFFINITY + 0.1, 0.0]])
    below = np.array([[0.0, NEUTRAL_AFFINITY - 0.1], [NEUTRAL_AFFINITY - 0.1, 0.0]])

    assert match_moment(above, views).groups == ((0, 1),)
    assert match_moment(below, views).groups == ()


def test_geometric_affinity_is_neutral_at_the_gate_and_zero_beyond_it():
    distance = np.array([0.0, 0.06, 0.12, 0.13, np.inf])

    affinity = geometric_affinity(distance, max_distance=0.12)

    np.testing.assert_allclose(affinity, [1.0, 0.75, NEUTRAL_AFFINITY, 0.0, 0.0])
