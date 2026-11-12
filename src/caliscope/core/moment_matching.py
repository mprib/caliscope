"""Match the boxes seen at one synchronized moment across cameras.

This is the multi-way matching of Dong et al. (2019), "Fast and Robust
Multi-Person 3D Pose Estimation from Multiple Views", Section 3.2. Every pair
of boxes in different cameras has an affinity in [0, 1]. The matching looks for
a low-rank, symmetric assignment that keeps high affinities, which makes the
matches agree around every loop of cameras. It is solved as a convex problem
with ADMM (their Algorithm 1), thresholded at 0.5, and closed transitively.

The solver settings are the ones in the authors' released code
(github.com/zju3dv/mvpose, Apache-2.0), since the paper does not print them.
The doubly stochastic projection is off there by default and is not included.

Three things differ from the paper and are Caliscope's own:

- The geometric affinity is a fixed function of the epipolar distance as a
  fraction of box size. The released code instead standardizes the pixel
  distances within each frame, which has no meaning when a frame holds one or
  two people.
- Dong's "appearance affinity" is called the re-identification affinity here.
  It comes from the cosine similarity of ReidentificationEmbeddings.

- When the thresholded matches close over two boxes of one view, those boxes
  are left unmatched at that moment and the rest of the group is kept.

How matches at single moments become associations of whole tracks is not part
of the published method. That lives in track_association.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Affinity at which a match neither helps nor hurts the objective (alpha_SVT in mvpose).
NEUTRAL_AFFINITY = 0.5
# Weight of the low-rank term (lambda_SVT in mvpose).
RANK_WEIGHT = 50.0
INITIAL_PENALTY = 64.0
TOLERANCE = 5e-4
MAX_ITERATIONS = 500
MATCH_THRESHOLD = 0.5


@dataclass(frozen=True)
class MomentMatch:
    """Groups of box indices matched at one moment.

    groups holds every group of two or more boxes with one box per view.
    inconsistent holds the boxes left out because the matching closed over two
    or more boxes of their view, which cannot be one object.
    """

    groups: tuple[tuple[int, ...], ...]
    inconsistent: tuple[tuple[int, ...], ...]


def geometric_affinity(distance: NDArray[np.float64], max_distance: float) -> NDArray[np.float64]:
    """Affinity from epipolar distance: 1 at zero, NEUTRAL_AFFINITY at max_distance, 0 beyond it or where not finite."""
    with np.errstate(invalid="ignore"):
        within = distance <= max_distance
    affinity = 1.0 - (1.0 - NEUTRAL_AFFINITY) * np.where(within, distance, 0.0) / max_distance
    return np.where(within, affinity, 0.0)


def fuse_affinities(geometric: NDArray[np.float64], reidentification: NDArray[np.float64]) -> NDArray[np.float64]:
    """Geometric mean of the two affinities (Dong, Eq. 2). It is 0 wherever the geometric affinity is 0."""
    return np.sqrt(geometric * reidentification)


def match_moment(affinity: NDArray[np.float64], views: NDArray[np.int64]) -> MomentMatch:
    """Match boxes across views from their pairwise affinities.

    affinity is (n, n) with values in [0, 1]; views gives each box's view.
    Entries between boxes of one view and on the diagonal are ignored.
    """
    n = len(views)
    different_view = views[:, None] != views[None, :]
    if n < 2 or not different_view.any():
        return MomentMatch((), ())

    relaxed = _solve(np.where(different_view, (affinity + affinity.T) / 2, 0.0), different_view)
    matched = (relaxed > MATCH_THRESHOLD) & different_view

    groups: list[tuple[int, ...]] = []
    inconsistent: list[tuple[int, ...]] = []
    for members in _connected(matched):
        if len(members) < 2:
            continue
        member_views = views[list(members)]
        repeated = np.isin(
            member_views, [view for view in set(member_views.tolist()) if (member_views == view).sum() > 1]
        )
        if not repeated.any():
            groups.append(members)
            continue
        # Two boxes of one view cannot be one object. Leave those boxes out and keep what still hangs together.
        inconsistent.append(tuple(int(i) for i in np.asarray(members)[repeated]))
        kept = np.asarray(members)[~repeated]
        for part in _connected(matched[np.ix_(kept, kept)]):
            if len(part) >= 2:
                groups.append(tuple(int(kept[i]) for i in part))
    return MomentMatch(tuple(groups), tuple(inconsistent))


def _solve(affinity: NDArray[np.float64], different_view: NDArray[np.bool_]) -> NDArray[np.float64]:
    """ADMM for min -<A, P> + lambda * ||P||_* over symmetric P in [0, 1], unit diagonal, no same-view matches."""
    n = len(affinity)
    cost = NEUTRAL_AFFINITY - affinity
    assignment = affinity.copy()
    dual = np.zeros_like(affinity)
    penalty = INITIAL_PENALTY
    diagonal = np.arange(n)

    for _ in range(MAX_ITERATIONS):
        previous = assignment
        u, singular_values, vt = np.linalg.svd(assignment + dual / penalty)
        low_rank = (u * np.maximum(singular_values - RANK_WEIGHT / penalty, 0.0)) @ vt

        assignment = np.clip(low_rank - (cost + dual) / penalty, 0.0, 1.0)
        assignment[~different_view] = 0.0
        assignment[diagonal, diagonal] = 1.0
        assignment = (assignment + assignment.T) / 2

        dual = dual + penalty * (assignment - low_rank)
        primal_residual = np.linalg.norm(assignment - low_rank) / n
        dual_residual = penalty * np.linalg.norm(assignment - previous) / n
        if primal_residual < TOLERANCE and dual_residual < TOLERANCE:
            break
        if primal_residual > 10 * dual_residual:
            penalty *= 2
        elif dual_residual > 10 * primal_residual:
            penalty /= 2
    return assignment


def _connected(adjacency: NDArray[np.bool_]) -> list[tuple[int, ...]]:
    """Connected components of an undirected graph, each sorted, in order of their smallest member."""
    n = len(adjacency)
    seen = np.zeros(n, dtype=bool)
    components: list[tuple[int, ...]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack, members = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbour in np.flatnonzero(adjacency[node] & ~seen):
                seen[neighbour] = True
                stack.append(int(neighbour))
        components.append(tuple(sorted(members)))
    return components


def reidentification_affinity(
    similarity: NDArray[np.float64], neutral_similarity: float, full_similarity: float
) -> NDArray[np.float64]:
    """Affinity from the cosine similarity of two unit embeddings.

    Linear in the similarity: NEUTRAL_AFFINITY at neutral_similarity, 1 at
    full_similarity and above, 0 as far below neutral as full is above it.
    """
    half_width = full_similarity - neutral_similarity
    return np.clip(NEUTRAL_AFFINITY + (similarity - neutral_similarity) / (2 * half_width), 0.0, 1.0)
