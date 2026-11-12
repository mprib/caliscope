"""Essential-matrix bootstrap for 2D-only observations (no object geometry).

The PnP path (pose_network_builder) needs each observation's known 3D object
location (obj_loc) to resection a camera against the calibration target. Body
keypoints carry no such geometry -- the person is the moving "target", and only
its 2D projections are known. This module recovers the extrinsic rig from those
2D-2D correspondences alone via the essential matrix, producing a result correct
up to a 7-DOF similarity transform (arbitrary scale, rotation, translation),
ready for bundle adjustment.

Ported from monokin's rig_calibration.py (validated on Pose2Sim), adapted to
caliscope's ImagePoints / CameraArray types. The numerics and RANSAC behavior
are monokin's; the data plumbing is caliscope's.

Composition (the pairwise-scale problem). Each essential pair is metric only up
to its own unknown baseline, so pairwise poses cannot simply be chained.
Instead one pair (the SCAFFOLD) is triangulated into a single 3D cloud in the
anchor camera's frame, and every other camera is registered to that one cloud by
resection. This yields a globally consistent rig in one shot, no loop closure --
cameras that share no frames with each other still both see the cloud.

Coplanarity (the multi-frame requirement and the scaffold guard). Body keypoints
at capture distance have a depth-to-baseline ratio near the coplanarity
degeneracy for essential estimation. pooled_correspondences pools each pair's
matches across every shared frame, where the subject has moved, so the pooled 3D
points span a volume rather than a plane. Near the degeneracy the essential
estimate can still flip to a wrong-but-self-consistent solution that cheirality
alone accepts, so the scaffold is not chosen by cheirality: each candidate is
validated by how well its cloud explains the *other* cameras (resection
reprojection error), the classic third-view disambiguation. A per-pair E
singular-value ratio is logged as a weak conditioning hint (findEssentialMat
always returns a (sigma, sigma, 0) matrix, so it rarely fires); the third-view
resection score is the operative degeneracy guard.

Intrinsics precondition: the essential decomposition has no obj_loc anchor to
absorb focal error, so it requires real calibrated intrinsics. The blind
`f = width/2` fallback is geometrically fatal here; calibrate_extrinsics gates
on this before ever reaching the bootstrap.
"""

from __future__ import annotations

import logging
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.bootstrap_pose.paired_pose_network import PairedPoseNetwork
from caliscope.core.bootstrap_pose.pose_network_builder import estimate_pnp_paired_pose_network
from caliscope.core.bootstrap_pose.stereopairs import StereoPair
from caliscope.core.point_data import ImagePoints

logger = logging.getLogger(__name__)

RANSAC_THRESHOLD_PX = 3.0  # essential-matrix inlier gate, pixels (converted to normalized units per camera)
RANSAC_PROB = 0.999
MIN_RESECTION_POINTS = 50  # cloud points a camera must share to be resectioned
MIN_CORRESPONDENCES = 8  # an essential matrix needs 8 point correspondences
CONDITIONING_FLOOR = 0.5  # E singular-value ratio below this flags a near-degenerate (coplanar) pair
MAX_SCAFFOLD_CANDIDATES = 12  # cap third-view validation cost on large rigs (correct scaffolds rank high by cheirality)


def pooled_correspondences(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[NDArray, NDArray, NDArray]:
    """Matched observation pixels for one camera pair, pooled over every shared frame.

    A correspondence is a single (object_id, keypoint_id) seen at the same
    sync_index by both cameras -- one 3D point at one instant viewed by two
    static cameras, a valid epipolar correspondence. Pooling across frames (as
    the subject moves) just adds points, which is what breaks the coplanarity
    degeneracy. Non-finite pixels drop out: a correspondence needs both views.

    Returns (keys, pixels_a, pixels_b): keys is (N, 3) int
    [object_id, keypoint_id, sync_index]; pixels_a/pixels_b are (N, 2) pixel
    coordinates in cameras a and b, row-aligned with keys.
    """
    merged = pd.merge(
        df_a,
        df_b,
        on=["sync_index", "object_id", "keypoint_id"],
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 2)), np.empty((0, 2))

    keys = merged[["object_id", "keypoint_id", "sync_index"]].to_numpy(dtype=np.int64)
    pix_a = merged[["img_loc_x_a", "img_loc_y_a"]].to_numpy(dtype=np.float64)
    pix_b = merged[["img_loc_x_b", "img_loc_y_b"]].to_numpy(dtype=np.float64)

    finite = np.isfinite(pix_a).all(axis=1) & np.isfinite(pix_b).all(axis=1)
    return keys[finite], pix_a[finite], pix_b[finite]


def _essential_conditioning(essential: NDArray) -> float:
    """Ratio of the two largest singular values of E; ~1.0 when well-conditioned.

    An ideal essential matrix has singular values (sigma, sigma, 0). The ratio
    of the second to the first drops toward 0 as the point cloud approaches
    coplanarity and the estimate becomes ambiguous.
    """
    singular_values = np.linalg.svd(essential, compute_uv=False)
    if singular_values[0] < 1e-12:
        return 0.0
    return float(singular_values[1] / singular_values[0])


def recover_pair_pose(pixels_a: NDArray, pixels_b: NDArray, *, camera_a: CameraData, camera_b: CameraData) -> dict:
    """Essential-matrix relative pose of camera b w.r.t. camera a from matched pixels.

    Normalizes each camera's pixels through its own intrinsics (removing lens
    distortion), runs findEssentialMat (RANSAC) then recoverPose. Returns
    rotation (b's world-to-camera when a is the world origin), unit translation,
    the inlier statistics, the E conditioning ratio, and the normalized inlier
    points (kept so the scaffold pair can be triangulated without renormalizing).

    Raises:
        ValueError: if essential-matrix estimation fails or is degenerate.
    """
    assert camera_a.matrix is not None and camera_b.matrix is not None
    norm_a = camera_a.undistort_points(pixels_a, output="normalized").astype(np.float64)
    norm_b = camera_b.undistort_points(pixels_b, output="normalized").astype(np.float64)

    # One shared threshold from the mean focal assumes the two focals are near-equal.
    mean_focal = 0.5 * (camera_a.matrix[0, 0] + camera_b.matrix[0, 0])
    threshold = RANSAC_THRESHOLD_PX / mean_focal  # normalized units

    essential, mask = cv2.findEssentialMat(
        norm_a,
        norm_b,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=RANSAC_PROB,
        threshold=threshold,
    )
    if essential is None or essential.shape != (3, 3):
        # findEssentialMat returns None on failure and a (3N, 3) stack of
        # candidate solutions on degenerate input; neither can be recovered.
        raise ValueError(
            "essential-matrix estimation failed or was degenerate "
            f"(got {'None' if essential is None else essential.shape})"
        )
    conditioning = _essential_conditioning(essential)
    mask = mask.ravel().astype(bool)
    _, rotation, translation, pose_mask = cv2.recoverPose(essential, norm_a[mask], norm_b[mask], cameraMatrix=np.eye(3))
    pose_mask = pose_mask.ravel().astype(bool)
    inlier_index = np.flatnonzero(mask)[pose_mask]  # rows of the original arrays kept by both stages
    return {
        "rotation": rotation,
        "translation": translation.ravel(),
        "inlier_fraction": float(mask.sum() / len(mask)),
        "n_inliers": int(mask.sum()),
        "n_total": int(len(mask)),
        "cheirality_inliers": int(pose_mask.sum()),
        "conditioning": conditioning,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "inlier_index": inlier_index,
    }


def triangulate_scaffold(pair_pose: dict, keys: NDArray) -> dict[tuple[int, int, int], NDArray]:
    """3D cloud from the scaffold pair, in camera a's (== world) frame at baseline 1.

    Triangulates the pair's cheirality-inlier normalized points with P_a = [I|0],
    P_b = [R|t]. Returns {(object_id, keypoint_id, sync_index): xyz} -- the anchor
    the other cameras resection against.
    """
    index = pair_pose["inlier_index"]
    norm_a = pair_pose["norm_a"][index]
    norm_b = pair_pose["norm_b"][index]
    projection_a = np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = np.hstack([pair_pose["rotation"], pair_pose["translation"].reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(projection_a, projection_b, norm_a.T, norm_b.T)
    weights = homogeneous[3]
    finite = np.abs(weights) > 1e-12  # w ~ 0: point at infinity, unusable 3D
    points = (homogeneous[:3, finite] / weights[finite]).T
    return {(int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])): points[j] for j, i in enumerate(index[finite])}


def resection_camera(
    cloud: dict[tuple[int, int, int], NDArray],
    df_cam: pd.DataFrame,
    camera: CameraData,
) -> tuple[NDArray, NDArray, int, float]:
    """World-to-camera pose of one camera by resection against the scaffold cloud.

    Gathers cloud points whose (object_id, keypoint_id, sync_index) this camera
    also observed with a finite pixel, then solvePnPRansac against its normalized
    pixels (K = identity). The cloud's 3D is pure 2D-2D triangulation, so no
    external depth enters the pose. The returned median reprojection error (in
    normalized units, over every matched point) measures how well this third view
    agrees with the cloud -- the signal that discriminates a correct scaffold from
    a geometrically wrong (twisted-pair) one.

    Raises:
        ValueError: if fewer than MIN_RESECTION_POINTS shared points exist, or
            solvePnPRansac fails.
    """
    assert camera.matrix is not None
    if not cloud:
        raise ValueError("scaffold cloud is empty")

    cloud_df = pd.DataFrame(
        [(o, k, s, xyz[0], xyz[1], xyz[2]) for (o, k, s), xyz in cloud.items()],
        columns=["object_id", "keypoint_id", "sync_index", "X", "Y", "Z"],
    )
    merged = pd.merge(df_cam, cloud_df, on=["object_id", "keypoint_id", "sync_index"])
    pixels = merged[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float64)
    finite = np.isfinite(pixels).all(axis=1)
    merged, pixels = merged[finite], pixels[finite]

    if len(merged) < MIN_RESECTION_POINTS:
        raise ValueError(f"only {len(merged)} cloud points to resection against")

    object_points = merged[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    normalized = camera.undistort_points(pixels, output="normalized").astype(np.float64)
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        object_points,
        normalized,
        np.eye(3),
        None,
        reprojectionError=RANSAC_THRESHOLD_PX / camera.matrix[0, 0],
        iterationsCount=200,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnPRansac failed")
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, np.eye(3), np.zeros(5))
    reproj_error = float(np.median(np.linalg.norm(normalized - projected.reshape(-1, 2), axis=1)))
    return rotation, tvec.ravel(), len(object_points), reproj_error


def _assemble_from_scaffold(
    scaffold_pair: tuple[int, int],
    scaffold_pose: dict,
    scaffold_keys: NDArray,
    cam_ids: list[int],
    per_cam: dict[int, pd.DataFrame],
    camera_array: CameraArray,
) -> tuple[dict[int, tuple[NDArray, NDArray]], tuple[int, float, int]]:
    """Build the full pose set from one candidate scaffold and score its global fit.

    Triangulates the scaffold cloud (anchor = scaffold_pair[0], at the origin),
    resections every other camera against it, and returns the poses together with
    a consistency score. The score is (n_failures, worst third-view reprojection
    error, -total cheirality inliers): a wrong scaffold reconstructs a cloud that
    third views cannot fit, inflating the reprojection error, so the correct
    scaffold minimizes this key. Cameras that cannot be resectioned (too little
    overlap with this cloud) count as failures but do not abort the assembly.
    """
    anchor_cam, other_cam = scaffold_pair
    cloud = triangulate_scaffold(scaffold_pose, scaffold_keys)
    poses: dict[int, tuple[NDArray, NDArray]] = {
        anchor_cam: (np.eye(3), np.zeros(3)),
        other_cam: (scaffold_pose["rotation"], scaffold_pose["translation"]),
    }
    reproj_errors: list[float] = []
    n_failures = 0
    for cam_id in cam_ids:
        if cam_id in poses:
            continue
        try:
            rotation, translation, _, reproj_error = resection_camera(
                cloud, per_cam[cam_id], camera_array.cameras[cam_id]
            )
        except ValueError:
            n_failures += 1
            continue
        poses[cam_id] = (rotation, translation)
        reproj_errors.append(reproj_error)

    worst_reproj = max(reproj_errors) if reproj_errors else 0.0
    score = (n_failures, worst_reproj, -scaffold_pose["cheirality_inliers"])
    return poses, score


def build_epipolar_pose_network(
    image_points: ImagePoints,
    camera_array: CameraArray,
) -> PairedPoseNetwork:
    """Recover the extrinsic rig from 2D-2D correspondences (no object geometry).

    For each candidate scaffold pair, triangulates its cloud in the pair's anchor
    frame and resections every other camera into it; the scaffold whose cloud the
    third views best agree with (lowest reprojection residual) wins. This third-
    view validation rejects wrong-but-self-consistent essential estimates that
    slip past cheirality alone near the coplanarity degeneracy. The winning rig is
    packaged as anchor-relative StereoPairs in a PairedPoseNetwork -- the same
    abstraction the PnP path produces, so bootstrap's apply_to + triangulate flow
    is unchanged. Baseline scale is arbitrary (fixed later by a metric scale cue).

    With exactly 2 cameras there is no third view to validate against, so the
    single essential pair is taken as-is (its twisted-pair ambiguity, if any, is
    fundamentally unresolvable without more views).

    Raises:
        CalibrationError: if fewer than 2 cameras have data, or no camera pair
            reaches the shared-correspondence minimum an essential matrix needs.
    """
    from caliscope.exceptions import CalibrationError

    df = image_points.df
    observed_cam_ids = set(df["cam_id"].unique())
    cam_ids = sorted(
        cam_id for cam_id, cam in camera_array.cameras.items() if not cam.ignore and cam_id in observed_cam_ids
    )
    if len(cam_ids) < 2:
        raise CalibrationError(f"Epipolar bootstrap needs at least 2 cameras with observations, found {len(cam_ids)}.")

    per_cam = {cam_id: df[df["cam_id"] == cam_id] for cam_id in cam_ids}

    pair_poses: dict[tuple[int, int], dict] = {}
    pair_keys: dict[tuple[int, int], NDArray] = {}
    for cam_a, cam_b in combinations(cam_ids, 2):
        keys, pix_a, pix_b = pooled_correspondences(per_cam[cam_a], per_cam[cam_b])
        if len(keys) < MIN_CORRESPONDENCES:
            continue
        try:
            pose = recover_pair_pose(
                pix_a,
                pix_b,
                camera_a=camera_array.cameras[cam_a],
                camera_b=camera_array.cameras[cam_b],
            )
        except ValueError as exc:
            logger.warning(f"Pair {cam_a}-{cam_b}: essential-matrix recovery failed ({exc})")
            continue
        pair_poses[(cam_a, cam_b)] = pose
        pair_keys[(cam_a, cam_b)] = keys
        logger.info(
            f"Pair {cam_a}-{cam_b}: {pose['n_inliers']}/{pose['n_total']} inliers, "
            f"{pose['cheirality_inliers']} cheirality, E conditioning {pose['conditioning']:.3f}"
        )
        if pose["conditioning"] < CONDITIONING_FLOOR:
            logger.warning(
                f"Pair {cam_a}-{cam_b}: essential matrix poorly conditioned "
                f"(singular-value ratio {pose['conditioning']:.3f} < {CONDITIONING_FLOOR})."
            )

    if not pair_poses:
        raise CalibrationError(
            f"Insufficient camera overlap for epipolar bootstrap: no camera pair reached the "
            f"{MIN_CORRESPONDENCES} shared correspondences an essential matrix needs. Cameras must "
            f"share observations of the moving subject across frames."
        )

    # Consider the strongest pairs (by cheirality) as scaffold candidates, and pick
    # the one whose cloud the other cameras best agree with. Third-view validation
    # is what catches a wrong essential estimate that cheirality alone accepts.
    candidates = sorted(pair_poses, key=lambda p: pair_poses[p]["cheirality_inliers"], reverse=True)
    candidates = candidates[:MAX_SCAFFOLD_CANDIDATES]

    best_poses: dict[int, tuple[NDArray, NDArray]] | None = None
    best_score: tuple[int, float, int] | None = None
    best_pair: tuple[int, int] | None = None
    for pair in candidates:
        poses, score = _assemble_from_scaffold(pair, pair_poses[pair], pair_keys[pair], cam_ids, per_cam, camera_array)
        if best_score is None or score < best_score:
            best_poses, best_score, best_pair = poses, score, pair

    assert best_poses is not None and best_pair is not None and best_score is not None
    anchor_cam = best_pair[0]
    logger.info(
        f"Selected scaffold {best_pair[0]}-{best_pair[1]} "
        f"(failures={best_score[0]}, worst third-view reprojection={best_score[1]:.5f}); "
        f"posed {len(best_poses)}/{len(cam_ids)} cameras, anchor = cam {anchor_cam}"
    )

    # Package as anchor-relative StereoPairs (primary < secondary, matching the
    # PnP path's convention so the shared RMSE/graph machinery keys correctly).
    aggregated: dict[tuple[int, int], StereoPair] = {}
    for cam_id, (rotation, translation) in best_poses.items():
        if cam_id == anchor_cam:
            continue
        pair = StereoPair(
            primary_cam_id=anchor_cam,
            secondary_cam_id=cam_id,
            error_score=float("nan"),
            rotation=rotation,
            translation=translation,
        )
        if pair.primary_cam_id > pair.secondary_cam_id:
            pair = pair.inverted()
        aggregated[pair.pair] = pair

    return estimate_pnp_paired_pose_network(aggregated, camera_array, image_points)
