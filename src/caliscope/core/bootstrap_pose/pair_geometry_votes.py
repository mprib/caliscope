"""Camera-pair geometries from camera-local tracks, without extrinsics.

Propose: each pairing of one track in camera a with one track in camera b fits an essential matrix on the
even-ranked shared moments, and keeps it only if the odd-ranked moments agree.
Vote: every other pairing between the same two cameras scores each proposal on its own odd-ranked moments, and
the proposal's matching is the greedy one-to-one set of agreeing pairings.
Close: over camera triplets, rotations, translations and matchings of one proposal per pair must agree, and
the proposals that close pick one geometry per camera pair.

Conventions. R_xy maps camera x's frame to camera y's, t_xy is the unit translation, so X_y = R_xy X_x + t_xy
and E_xy = [t_xy]x R_xy satisfies x_y^T E_xy x_x = 0 on normalized coordinates.
The line angle between two rays with unit directions d_i, d_j is arccos(|d_i . d_j|), from 0 degrees (one
line) to 90 degrees.
Sampson distance is converted to pixels with the mean focal length of the two cameras.
Rotation closure is angle(R_ac^T R_bc R_ab). Translation closure is asin(|det[u1, u2, u3]|) for the baselines
u1 = -R_ab^T t_ab, u2 = -R_ac^T t_ac, u3 = R_ab^T (-R_bc^T t_bc) in camera a's frame, which are coplanar for
consistent geometries.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.bootstrap_pose.epipolar_pose_builder import _recover_logged
from caliscope.core.point_data import ImagePoints
from caliscope.core.track_association import (
    MIN_INLIER_FRACTION,
    MIN_KEYPOINTS_PER_MOMENT,
    MIN_RAY_ANGLE_DEG,
    MIN_SUPPORT_MOMENTS,
    _Rig,
    _TrackKey,
    _tracks_by_camera,
    _unit_rays,
)

logger = logging.getLogger(__name__)

# Placeholder values, pending measurement over the synthetic scenes.
# Shared moments a track pairing needs to propose a geometry or to vote.
MIN_PROPOSAL_MOMENTS = 8
# Sampson distance in pixels under which a correspondence agrees with a geometry.
VOTE_THRESHOLD_PX = 4.0
# Pixel cell that makes a repeated keypoint position count once in a vote's weight.
SUPPORT_CELL_PX = 16
# Proposals voted on per camera pair, as a multiple of the larger track count of the two cameras.
PROPOSALS_VOTED_PER_TRACK = 2
# Proposals kept per camera pair after merging near duplicates.
KEPT_PROPOSALS_PER_PAIR = 3
# Two proposals within this rotation and translation-direction angle describe one geometry.
SAME_GEOMETRY_DEG = 2.0
# Largest rotation closure angle over a camera triplet.
ROTATION_CLOSURE_DEG = 2.0
# Largest translation closure angle over a camera triplet.
TRANSLATION_CLOSURE_DEG = 2.0

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected by closure"
STATUS_UNCLOSED = "unclosed"
STATUS_OUTSIDE_COMPONENT = "outside the largest component"

_CamPair = tuple[int, int]


@dataclass(frozen=True, eq=False)
class _PixelTrack:
    """One track's observations, sorted by a shared (sync_index, keypoint_id) key."""

    key: _TrackKey
    point_keys: NDArray[np.int64]
    sync_indices: NDArray[np.int64]
    keypoint_ids: NDArray[np.int64]
    pixels: NDArray[np.float64]
    xy: NDArray[np.float64]  # normalized image coordinates
    moments: frozenset[int]


@dataclass(frozen=True, eq=False)
class _Correspondences:
    """Keypoints seen by both tracks of a pairing at the same moment, row-aligned."""

    sync_indices: NDArray[np.int64]
    keypoint_ids: NDArray[np.int64]
    pixels_a: NDArray[np.float64]
    pixels_b: NDArray[np.float64]
    xy_a: NDArray[np.float64]
    xy_b: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.sync_indices)

    def select(self, mask: NDArray[np.bool_]) -> _Correspondences:
        return _Correspondences(
            self.sync_indices[mask],
            self.keypoint_ids[mask],
            self.pixels_a[mask],
            self.pixels_b[mask],
            self.xy_a[mask],
            self.xy_b[mask],
        )


@dataclass(frozen=True, eq=False)
class _Pairing:
    """A track in camera a with a track in camera b, split into fit and held-out correspondences."""

    key_a: _TrackKey
    key_b: _TrackKey
    fit: _Correspondences
    held_out: _Correspondences


@dataclass(frozen=True, eq=False)
class _Proposal:
    """A pair geometry fitted on one pairing's fit half. Camera a is the origin."""

    cam_a: int
    cam_b: int
    track_a: int
    track_b: int
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]  # unit length
    n_fit_inliers: int
    conditioning: float
    held_out_inliers: NDArray[np.bool_]

    @property
    def pairing(self) -> tuple[int, int]:
        return (self.track_a, self.track_b)

    @property
    def essential(self) -> NDArray[np.float64]:
        return _essential(self.rotation, self.translation)


@dataclass(frozen=True, eq=False)
class _Proposals:
    pairings: dict[_CamPair, list[_Pairing]]
    proposals: dict[_CamPair, list[_Proposal]]


@dataclass(frozen=True, eq=False)
class _VotedProposal:
    """A proposal with its vote, its matching of track pairs, and its lead over the next distinct geometry."""

    proposal: _Proposal
    vote: float
    matching: tuple[tuple[int, int], ...]  # (track_a, track_b), one-to-one
    margin: float  # vote minus the next kept geometry's vote; the vote itself when it is the only one

    @property
    def n_voters(self) -> int:
        return len(self.matching)


@dataclass(frozen=True)
class _TripletCheck:
    rotation_deg: float
    translation_deg: float
    contradictions: int

    @property
    def consistent(self) -> bool:
        return (
            self.rotation_deg <= ROTATION_CLOSURE_DEG
            and self.translation_deg <= TRANSLATION_CLOSURE_DEG
            and self.contradictions == 0
        )


@dataclass(frozen=True, eq=False)
class _PairDecision:
    """The geometry chosen for one camera pair and what closure said about it."""

    cam_a: int
    cam_b: int
    chosen: _VotedProposal
    status: str
    accepted: bool  # in the largest component of the accepted graph
    closure: int  # triplets in which a kept proposal like this one closes
    closure_score: float
    worst_rotation_deg: float  # NaN when the pair is in no complete triplet
    worst_translation_deg: float
    identity_contradictions: int


@dataclass(frozen=True, eq=False)
class _ClosureResult:
    decisions: dict[_CamPair, _PairDecision]
    kept: dict[_CamPair, tuple[_VotedProposal, ...]]
    placed_cameras: tuple[int, ...]
    unplaced_cameras: tuple[int, ...]

    @property
    def accepted(self) -> dict[_CamPair, _VotedProposal]:
        return {pair: decision.chosen for pair, decision in self.decisions.items() if decision.accepted}


###############################################################################
# Indexing
###############################################################################


def _index_pixel_tracks(points: ImagePoints, cameras: CameraArray) -> dict[_TrackKey, _PixelTrack]:
    """Split unlinked points into per-track arrays of pixels and undistorted normalized coordinates."""
    df = points.df
    finite = np.isfinite(df[["img_loc_x", "img_loc_y"]].to_numpy(dtype=float)).all(axis=1)
    df = df[finite].drop_duplicates(subset=["cam_id", "track_id", "sync_index", "keypoint_id"]).reset_index(drop=True)
    point_key = df.groupby(["sync_index", "keypoint_id"], sort=True).ngroup().to_numpy(dtype=np.int64)

    pixels = df[["img_loc_x", "img_loc_y"]].to_numpy(dtype=float)
    normalized = np.empty_like(pixels)
    cam_column = df["cam_id"].to_numpy()
    for cam_id in sorted(set(cam_column.tolist())):
        mask = cam_column == cam_id
        normalized[mask] = cameras.cameras[int(cam_id)].undistort_points(pixels[mask], output="normalized")

    track_column = df["track_id"].to_numpy()
    sync_column = df["sync_index"].to_numpy(dtype=np.int64)
    keypoint_column = df["keypoint_id"].to_numpy(dtype=np.int64)
    tracks: dict[_TrackKey, _PixelTrack] = {}
    for _, rows in df.groupby(["cam_id", "track_id"], sort=True):
        positions = rows.index.to_numpy()
        positions = positions[np.argsort(point_key[positions], kind="stable")]
        key = (int(cam_column[positions[0]]), int(track_column[positions[0]]))
        tracks[key] = _PixelTrack(
            key=key,
            point_keys=point_key[positions],
            sync_indices=sync_column[positions],
            keypoint_ids=keypoint_column[positions],
            pixels=pixels[positions],
            xy=normalized[positions],
            moments=frozenset(int(s) for s in sync_column[positions]),
        )
    return tracks


def _pairing_of(track_a: _PixelTrack, track_b: _PixelTrack) -> _Pairing | None:
    """Split the pair's correspondences by shared-moment rank, or None when too few moments are shared."""
    if len(track_a.moments & track_b.moments) < MIN_PROPOSAL_MOMENTS:
        return None
    _, in_a, in_b = np.intersect1d(track_a.point_keys, track_b.point_keys, assume_unique=True, return_indices=True)
    shared = _Correspondences(
        track_a.sync_indices[in_a],
        track_a.keypoint_ids[in_a],
        track_a.pixels[in_a],
        track_b.pixels[in_b],
        track_a.xy[in_a],
        track_b.xy[in_b],
    )
    moments = np.unique(shared.sync_indices)
    if len(moments) < MIN_PROPOSAL_MOMENTS:
        return None
    in_fit = np.isin(shared.sync_indices, moments[0::2])
    return _Pairing(track_a.key, track_b.key, fit=shared.select(in_fit), held_out=shared.select(~in_fit))


###############################################################################
# Stage 1: propose
###############################################################################


def _propose_pair_geometries(tracks: Mapping[_TrackKey, _PixelTrack], cameras: CameraArray) -> _Proposals:
    """Fit an essential matrix per track pairing and keep those that hold on the pairing's held-out half."""
    by_camera = _tracks_by_camera(tracks)
    pairings: dict[_CamPair, list[_Pairing]] = {}
    proposals: dict[_CamPair, list[_Proposal]] = {}
    for cam_a, cam_b in combinations(sorted(by_camera), 2):
        focal = _mean_focal(cameras, cam_a, cam_b)
        pair_pairings: list[_Pairing] = []
        pair_proposals: list[_Proposal] = []
        n_discarded = 0
        for key_a in by_camera[cam_a]:
            for key_b in by_camera[cam_b]:
                pairing = _pairing_of(tracks[key_a], tracks[key_b])
                if pairing is None:
                    continue
                pair_pairings.append(pairing)
                pose = _recover_logged(
                    cam_a,
                    cam_b,
                    pairing.fit.pixels_a,
                    pairing.fit.pixels_b,
                    cameras,
                    f"Proposal {cam_a}:{key_a[1]}-{cam_b}:{key_b[1]}",
                )
                if pose is None:
                    continue
                rotation = np.asarray(pose["rotation"], dtype=float)
                translation = np.asarray(pose["translation"], dtype=float).reshape(3)
                distance = _sampson_px(
                    _essential(rotation, translation), pairing.held_out.xy_a, pairing.held_out.xy_b, focal
                )
                inliers = distance <= VOTE_THRESHOLD_PX
                if inliers.mean() < MIN_INLIER_FRACTION:
                    n_discarded += 1
                    continue
                pair_proposals.append(
                    _Proposal(
                        cam_a=cam_a,
                        cam_b=cam_b,
                        track_a=key_a[1],
                        track_b=key_b[1],
                        rotation=rotation,
                        translation=translation,
                        n_fit_inliers=int(pose["cheirality_inliers"]),
                        conditioning=float(pose["conditioning"]),
                        held_out_inliers=inliers,
                    )
                )
        if pair_pairings:
            pairings[(cam_a, cam_b)] = pair_pairings
        if pair_proposals:
            proposals[(cam_a, cam_b)] = pair_proposals
        logger.info(
            f"Cameras {cam_a}-{cam_b}: {len(pair_pairings)} pairings tried, {len(pair_proposals)} proposals, "
            f"{n_discarded} discarded by the held-out self-check"
        )
    return _Proposals(pairings=pairings, proposals=proposals)


def _essential(rotation: NDArray[np.float64], translation: NDArray[np.float64]) -> NDArray[np.float64]:
    """E = [t]x R, so that x_b^T E x_a = 0 for normalized coordinates of one point seen by cameras a and b."""
    t = translation
    skew = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    return skew @ rotation


def _mean_focal(cameras: CameraArray, cam_a: int, cam_b: int) -> float:
    matrix_a, matrix_b = cameras.cameras[cam_a].matrix, cameras.cameras[cam_b].matrix
    assert matrix_a is not None and matrix_b is not None
    return 0.5 * float(matrix_a[0, 0] + matrix_b[0, 0])


def _sampson_px(
    essential: NDArray[np.float64], xy_a: NDArray[np.float64], xy_b: NDArray[np.float64], focal_px: float
) -> NDArray[np.float64]:
    """Sampson distance of normalized correspondences under E, in pixels."""
    homogeneous_a = np.column_stack([xy_a, np.ones(len(xy_a))])
    homogeneous_b = np.column_stack([xy_b, np.ones(len(xy_b))])
    from_a = homogeneous_a @ essential.T
    from_b = homogeneous_b @ essential
    numerator = np.abs(np.einsum("ni,ni->n", homogeneous_b, from_a))
    denominator = np.sqrt(from_a[:, 0] ** 2 + from_a[:, 1] ** 2 + from_b[:, 0] ** 2 + from_b[:, 1] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(denominator > 0, numerator / denominator, np.inf)
    return focal_px * distance


###############################################################################
# Stage 2: vote
###############################################################################


@dataclass(frozen=True)
class _Evaluation:
    agrees: bool
    weight: float


def _evaluate_voter(proposal: _Proposal, pairing: _Pairing, focal_px: float) -> _Evaluation:
    """Score one pairing's held-out correspondences against a proposal's geometry."""
    held_out = pairing.held_out
    distance = _sampson_px(proposal.essential, held_out.xy_a, held_out.xy_b, focal_px)
    inlier = distance <= VOTE_THRESHOLD_PX
    if not len(inlier) or inlier.mean() < MIN_INLIER_FRACTION:
        return _Evaluation(agrees=False, weight=0.0)
    _, per_moment = np.unique(held_out.sync_indices[inlier], return_counts=True)
    if int((per_moment >= MIN_KEYPOINTS_PER_MOMENT).sum()) < MIN_SUPPORT_MOMENTS:
        return _Evaluation(agrees=False, weight=0.0)
    return _Evaluation(agrees=True, weight=_conditioned_support(proposal, held_out.select(inlier)))


def _conditioned_support(proposal: _Proposal, inliers: _Correspondences) -> float:
    """Distinct (keypoint, pixel cell) inliers, each weighted by how well its rays cross under the proposal."""
    n = len(inliers)
    projections = np.stack(
        [np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([proposal.rotation, proposal.translation.reshape(3, 1)])]
    )
    rig = _Rig(cam_ids=(proposal.cam_a, proposal.cam_b), projections=projections, focal_px=np.ones(2))
    rays_a = _unit_rays(np.zeros(n, dtype=np.int64), inliers.xy_a, rig)
    rays_b = _unit_rays(np.ones(n, dtype=np.int64), inliers.xy_b, rig)
    theta = np.arccos(np.clip(np.abs(np.einsum("ni,ni->n", rays_a, rays_b)), 0.0, 1.0))
    factor = np.minimum(1.0, np.sin(theta) / np.sin(np.radians(MIN_RAY_ANGLE_DEG)))

    cells = np.floor(inliers.pixels_a / SUPPORT_CELL_PX).astype(np.int64)
    _, group = np.unique(np.column_stack([inliers.keypoint_ids, cells]), axis=0, return_inverse=True)
    group = group.reshape(-1)
    best = np.zeros(int(group.max()) + 1 if n else 0)
    np.maximum.at(best, group, factor)
    return float(best.sum())


def _cast_votes(
    proposal: _Proposal, pairings: Sequence[_Pairing], focal_px: float
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Total weight and matching of the agreeing pairings, accepted strongest first with each track used once."""
    agreeing: list[tuple[float, int, int]] = []
    for pairing in pairings:
        evaluation = _evaluate_voter(proposal, pairing, focal_px)
        if evaluation.agrees:
            agreeing.append((-evaluation.weight, pairing.key_a[1], pairing.key_b[1]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matching: list[tuple[int, int]] = []
    total = 0.0
    for negative_weight, track_a, track_b in sorted(agreeing):
        if track_a in used_a or track_b in used_b:
            continue
        used_a.add(track_a)
        used_b.add(track_b)
        matching.append((track_a, track_b))
        total -= negative_weight
    return total, tuple(matching)


def _same_geometry(p: _Proposal, q: _Proposal) -> bool:
    rotation = _rotation_angle_deg(p.rotation.T @ q.rotation)
    direction = float(np.degrees(np.arccos(np.clip(p.translation @ q.translation, -1.0, 1.0))))
    return rotation < SAME_GEOMETRY_DEG and direction < SAME_GEOMETRY_DEG


def _rotation_angle_deg(rotation: NDArray[np.float64]) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def _vote(candidates: _Proposals, cameras: CameraArray) -> dict[_CamPair, list[_VotedProposal]]:
    """Per camera pair, the kept distinct geometries with their votes, matchings and margins, best first."""
    voted_by_pair: dict[_CamPair, list[_VotedProposal]] = {}
    for pair, proposals in sorted(candidates.proposals.items()):
        pairings = candidates.pairings[pair]
        focal = _mean_focal(cameras, *pair)
        by_tracks = {(pairing.key_a[1], pairing.key_b[1]): pairing for pairing in pairings}

        own_weight = {
            proposal.pairing: _evaluate_voter(proposal, by_tracks[proposal.pairing], focal).weight
            for proposal in proposals
        }
        n_tracks = max(len({p.key_a for p in pairings}), len({p.key_b for p in pairings}))
        limit = PROPOSALS_VOTED_PER_TRACK * n_tracks
        ranked = sorted(proposals, key=lambda p: (-own_weight[p.pairing], p.pairing))[:limit]

        scored: list[tuple[_Proposal, float, tuple[tuple[int, int], ...]]] = []
        for proposal in ranked:
            vote, matching = _cast_votes(proposal, pairings, focal)
            if matching:
                scored.append((proposal, vote, matching))
        scored.sort(key=lambda item: (-item[1], item[0].pairing))

        kept: list[tuple[_Proposal, float, tuple[tuple[int, int], ...]]] = []
        for item in scored:
            if any(_same_geometry(item[0], other[0]) for other in kept):
                continue
            kept.append(item)
            if len(kept) == KEPT_PROPOSALS_PER_PAIR:
                break
        if not kept:
            continue

        voted = [
            _VotedProposal(
                proposal=proposal,
                vote=vote,
                matching=matching,
                margin=vote - (kept[index + 1][1] if index + 1 < len(kept) else 0.0),
            )
            for index, (proposal, vote, matching) in enumerate(kept)
        ]
        voted_by_pair[pair] = voted
        summary = ", ".join(
            f"{v.proposal.track_a}:{v.proposal.track_b} vote {v.vote:.1f} voters {v.n_voters} margin {v.margin:.1f}"
            for v in voted
        )
        logger.info(f"Cameras {pair[0]}-{pair[1]}: kept {summary}")
    return voted_by_pair


###############################################################################
# Stage 3: close over camera triplets
###############################################################################


def _check_triplet(
    p_ab: _VotedProposal,
    p_bc: _VotedProposal,
    p_ac: _VotedProposal,
    track_moments: Mapping[_TrackKey, frozenset[int]],
) -> _TripletCheck:
    ab, bc, ac = p_ab.proposal, p_bc.proposal, p_ac.proposal
    rotation_deg = _rotation_angle_deg(ac.rotation.T @ bc.rotation @ ab.rotation)

    u1 = -ab.rotation.T @ ab.translation
    u2 = -ac.rotation.T @ ac.translation
    u3 = ab.rotation.T @ (-bc.rotation.T @ bc.translation)
    determinant = float(np.linalg.det(np.column_stack([u1, u2, u3])))
    translation_deg = float(np.degrees(np.arcsin(min(1.0, abs(determinant)))))

    to_b = dict(p_ab.matching)
    to_c = dict(p_bc.matching)
    direct = dict(p_ac.matching)
    cam_c = ac.cam_b
    contradictions = 0
    for track_a, track_b in to_b.items():
        track_c = to_c.get(track_b)
        other = direct.get(track_a)
        if track_c is None or other is None or other == track_c:
            continue
        if track_moments.get((cam_c, other), frozenset()) & track_moments.get((cam_c, track_c), frozenset()):
            contradictions += 1
    return _TripletCheck(rotation_deg, translation_deg, contradictions)


def _close_triplets(
    voted: Mapping[_CamPair, Sequence[_VotedProposal]],
    track_moments: Mapping[_TrackKey, frozenset[int]],
    cam_ids: Sequence[int],
) -> _ClosureResult:
    """Choose one geometry per camera pair by closure over triplets and keep the largest connected component.

    With fewer than three cameras there are no triplets and the top-voted geometry is accepted.
    """
    cam_ids = sorted(cam_ids)
    triplets = [
        trio
        for trio in combinations(cam_ids, 3)
        if all(pair in voted for pair in ((trio[0], trio[1]), (trio[1], trio[2]), (trio[0], trio[2])))
    ]

    checks: dict[tuple[tuple[int, int, int], tuple[int, ...]], _TripletCheck] = {}
    closure: dict[tuple[_CamPair, int], list[float]] = {
        (pair, index): [0.0, 0.0] for pair in voted for index in range(len(voted[pair]))
    }
    for trio in triplets:
        a, b, c = trio
        pairs = ((a, b), (b, c), (a, c))
        best: dict[tuple[_CamPair, int], float] = {}
        for indices in product(*(range(len(voted[pair])) for pair in pairs)):
            members = [voted[pair][index] for pair, index in zip(pairs, indices)]
            check = _check_triplet(members[0], members[1], members[2], track_moments)
            checks[(trio, indices)] = check
            if check.consistent:
                score = sum(member.vote for member in members)
                for pair, index in zip(pairs, indices):
                    best[(pair, index)] = max(best.get((pair, index), 0.0), score)
        for key, score in best.items():
            closure[key][0] += 1
            closure[key][1] = max(closure[key][1], score)

    chosen_index: dict[_CamPair, int] = {}
    for pair, proposals in voted.items():
        chosen_index[pair] = min(
            range(len(proposals)),
            key=lambda i: (
                -closure[(pair, i)][0],
                -closure[(pair, i)][1],
                -proposals[i].vote,
                proposals[i].proposal.pairing,
            ),
        )

    in_trios: dict[_CamPair, list[_TripletCheck]] = {pair: [] for pair in voted}
    for trio in triplets:
        a, b, c = trio
        pairs = ((a, b), (b, c), (a, c))
        check = checks[(trio, (chosen_index[(a, b)], chosen_index[(b, c)], chosen_index[(a, c)]))]
        for pair in pairs:
            in_trios[pair].append(check)

    status: dict[_CamPair, str] = {}
    for pair in voted:
        trio_checks = in_trios[pair]
        if len(cam_ids) < 3:
            status[pair] = STATUS_ACCEPTED
        elif not trio_checks:
            status[pair] = STATUS_UNCLOSED
        elif any(check.consistent for check in trio_checks):
            status[pair] = STATUS_ACCEPTED
        else:
            status[pair] = STATUS_REJECTED

    accepted = {pair for pair, value in status.items() if value == STATUS_ACCEPTED}
    covered = {cam for pair in accepted for cam in pair}
    unclosed = sorted(
        (pair for pair, value in status.items() if value == STATUS_UNCLOSED),
        key=lambda pair: (-voted[pair][chosen_index[pair]].vote, pair),
    )
    for pair in unclosed:
        if pair[0] not in covered or pair[1] not in covered:
            accepted.add(pair)
            covered |= set(pair)

    placed = _largest_component(accepted)
    decisions: dict[_CamPair, _PairDecision] = {}
    for pair in sorted(voted):
        trio_checks = in_trios[pair]
        index = chosen_index[pair]
        is_placed = pair in accepted and pair[0] in placed
        pair_status = status[pair]
        if pair in accepted and not is_placed:
            pair_status = STATUS_OUTSIDE_COMPONENT
        decisions[pair] = _PairDecision(
            cam_a=pair[0],
            cam_b=pair[1],
            chosen=voted[pair][index],
            status=pair_status,
            accepted=is_placed,
            closure=int(closure[(pair, index)][0]),
            closure_score=closure[(pair, index)][1],
            worst_rotation_deg=max((c.rotation_deg for c in trio_checks), default=float("nan")),
            worst_translation_deg=max((c.translation_deg for c in trio_checks), default=float("nan")),
            identity_contradictions=sum(c.contradictions for c in trio_checks),
        )

    placed_cameras = tuple(sorted(placed))
    unplaced_cameras = tuple(cam for cam in cam_ids if cam not in placed)
    logger.info(
        f"Closure over {len(triplets)} triplets: accepted {sorted(p for p, d in decisions.items() if d.accepted)}, "
        f"unplaced cameras {list(unplaced_cameras)}"
    )
    return _ClosureResult(
        decisions=decisions,
        kept={pair: tuple(proposals) for pair, proposals in voted.items()},
        placed_cameras=placed_cameras,
        unplaced_cameras=unplaced_cameras,
    )


def _largest_component(pairs: set[_CamPair]) -> set[int]:
    """Cameras of the connected component with the most cameras; ties go to the smallest cam_id."""
    neighbours: dict[int, set[int]] = {}
    for a, b in pairs:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    components: list[set[int]] = []
    seen: set[int] = set()
    for start in sorted(neighbours):
        if start in seen:
            continue
        component = {start}
        frontier = [start]
        while frontier:
            for neighbour in neighbours[frontier.pop()] - component:
                component.add(neighbour)
                frontier.append(neighbour)
        seen |= component
        components.append(component)
    return min(components, key=lambda component: (-len(component), min(component)), default=set())
