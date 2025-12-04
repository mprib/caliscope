# caliscope/calibration/array_initialization/pose_network_builder.py

from itertools import combinations
from numpy.typing import NDArray
import cv2
import time
import numpy as np
import pandas as pd
import logging
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from caliscope.calibration.array_initialization.paired_pose_network import PairedPoseNetwork
from caliscope.calibration.array_initialization.stereopairs import StereoPair
from caliscope.cameras.camera_array import CameraArray
from caliscope.post_processing.point_data import ImagePoints

logger = logging.getLogger(__file__)

# Add these near the top with other constants
DEFAULT_MIN_PNP_POINTS = 4
DEFAULT_OUTLIER_THRESHOLD = 1.5
DEFAULT_GOLD_STANDARD_BOARDS = 10


class PoseNetworkBuilder:
    """
    Fluent builder for creating PairedPoseNetwork from camera array and point data.

    This class encapsulates the multi-stage PnP pose estimation pipeline:
    1. Camera-to-object pose estimation
    2. Relative pose computation
    3. Outlier filtering
    4. Aggregation and network creation

    Example:
        builder = PoseNetworkBuilder(camera_array, point_data)
        network = (
            builder
            .estimate_camera_to_object_poses(min_points=6)
            .estimate_relative_poses()
            .filter_outliers(threshold=1.5)
            .build()
        )
        network.apply_to(camera_array)
    """

    def __init__(self, camera_array: CameraArray, image_points: ImagePoints):
        """
        Initialize the builder.

        Args:
            camera_array: Camera array with intrinsic calibration
            point_data: DataFrame with 2D/3D point correspondences
                       Required columns: sync_index, port, point_id,
                                       img_loc_x, img_loc_y, obj_loc_x, obj_loc_y
        """
        self.camera_array: CameraArray = camera_array
        self._image_points: ImagePoints = image_points

        # Pipeline state
        self._camera_to_object_poses: (
            dict[tuple[int, int], tuple[NDArray[np.float64], NDArray[np.float64], float]] | None
        ) = None
        self._relative_poses: dict[tuple[tuple[int, int], int], StereoPair] | None = None
        self._filtered_poses: dict[tuple[int, int], list[StereoPair]] | None = None
        self._aggregated_poses: dict[tuple[int, int], StereoPair] | None = None
        self._pnp_network: PairedPoseNetwork | None = None

        # Simple state tracking, no error accumulation
        self._state: str = "initialized"

    @property
    def state(self) -> str:
        """Current pipeline state (for inspection/debugging)."""
        return self._state

    def estimate_camera_to_object_poses(
        self,
        min_points: int = DEFAULT_MIN_PNP_POINTS,
        pnp_flags: int = cv2.SOLVEPNP_IPPE,
        fallback_flags: int = cv2.SOLVEPNP_ITERATIVE,
    ) -> Self:
        """
        Step 1a: Estimate per-camera poses relative to object coordinate frame.

        Args:
            min_points: Minimum points for reliable PnP (default: 4)
            pnp_flags: Primary PnP algorithm (default: SOLVEPNP_IPPE)
            fallback_flags: Fallback algorithm (default: SOLVEPNP_ITERATIVE)

        Returns:
            self for method chaining
        """
        logger.info("=" * 20 + " Step 1a: Camera-to-Object Poses " + "=" * 20)

        # Clear all downstream state when re-running step 1a
        self._camera_to_object_poses = None
        self._relative_poses = None
        self._filtered_poses = None
        self._aggregated_poses = None
        self._pnp_network = None

        try:
            self._camera_to_object_poses = compute_camera_to_object_poses_pnp(
                self._image_points,
                self.camera_array,
                min_points=min_points,
                pnp_flags=pnp_flags,
                fallback_flags=fallback_flags,
            )
            self._state = "camera_poses_estimated"
        except Exception as e:
            logger.error(f"Failed to estimate camera-to-object poses: {e}")
            raise

        return self

    def estimate_relative_poses(self) -> Self:
        """
        Step 1b: Compute relative poses between camera pairs.

        Requires: estimate_camera_to_object_poses() must be called first.

        Returns:
            self for method chaining
        """
        logger.info("=" * 20 + " Step 1b: Relative Poses " + "=" * 20)

        if self._camera_to_object_poses is None:
            raise RuntimeError("Must call estimate_camera_to_object_poses() first")

        try:
            self._relative_poses = compute_relative_poses(self._camera_to_object_poses, self.camera_array)
            self._state = "relative_poses_estimated"
        except Exception as e:
            logger.error(f"Failed to estimate relative poses: {e}")
            raise

        return self

    def filter_outliers(
        self,
        threshold: float = DEFAULT_OUTLIER_THRESHOLD,
        rotation_threshold_multiplier: float | None = None,
        translation_threshold_multiplier: float | None = None,
    ) -> Self:
        """
        Step 2: Apply IQR-based outlier rejection to relative poses.

        Args:
            threshold: IQR multiplier for outlier detection (default: 1.5)
            rotation_threshold_multiplier: Optional separate IQR multiplier for rotation outliers
            translation_threshold_multiplier: Optional separate IQR multiplier for translation outliers

        Requires: estimate_relative_poses() must be called first.

        Returns:
            self for method chaining
        """
        logger.info("=" * 20 + " Step 2: Outlier Filtering " + "=" * 20)

        if self._relative_poses is None:
            raise RuntimeError("Must call estimate_relative_poses() first")

        try:
            self._filtered_poses = reject_outliers(
                self._relative_poses,
                threshold=threshold,
                rotation_threshold_multiplier=rotation_threshold_multiplier,
                translation_threshold_multiplier=translation_threshold_multiplier,
            )
            self._state = "filtered"
        except Exception as e:
            logger.error(f"Failed to filter outliers: {e}")
            raise

        return self

    def build(self) -> PairedPoseNetwork:
        """
        Step 3: Aggregate poses and create PairedPoseNetwork with RMSE scores.

        Requires: filter_outliers() must be called first.

        Returns:
            Configured PairedPoseNetwork ready for application to camera array
        """
        logger.info("=" * 20 + " Step 3: Build PairedPoseNetwork " + "=" * 20)

        if self._filtered_poses is None:
            raise RuntimeError("Must call filter_outliers() first")

        try:
            self._aggregated_poses = aggregate_poses(self._filtered_poses)
            self._pnp_network = estimate_pnp_paired_pose_network(
                self._aggregated_poses, self.camera_array, self._image_points
            )
            self._state = "built"
            logger.info(f"Successfully built PairedPoseNetwork with {len(self._pnp_network._pairs)} pairs")
            return self._pnp_network
        except Exception as e:
            logger.error(f"Failed to build PairedPoseNetwork: {e}")
            raise


def compute_camera_to_object_poses_pnp(
    image_points: ImagePoints,
    camera_array: CameraArray,
    min_points: int = DEFAULT_MIN_PNP_POINTS,
    pnp_flags: int = cv2.SOLVEPNP_IPPE,
    fallback_flags: int = cv2.SOLVEPNP_ITERATIVE,
) -> dict[tuple[int, int], tuple[NDArray[np.float64], NDArray[np.float64], float]]:
    """
    Compute per-camera poses relative to the object coordinate frame for each sync_index.

    Args:
        image_points: Validated 2D point correspondences
        camera_array: Camera array with intrinsic parameters
        min_points: Minimum points required for reliable PnP (default: 4)
        pnp_flags: Primary PnP algorithm flag (default: SOLVEPNP_IPPE for planar targets)
        fallback_flags: Fallback algorithm if primary fails (default: SOLVEPNP_ITERATIVE)

    Returns:
        dict mapping (port, sync_index) -> (R, t, reprojection_error)
    """
    logger.info(f"Computing per-frame camera poses with PnP (min_points={min_points})...")

    # Pre-undistort all points per camera
    logger.info("Pre-undistorting points...")
    undistorted_data = []
    for port, camera in camera_array.cameras.items():
        if camera.matrix is None:
            logger.warning(f"Camera {port} missing intrinsics, skipping")
            continue

        cam_data = image_points.df[image_points.df["port"] == port].copy()
        if cam_data.empty:
            continue

        img_points = cam_data[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32)
        undistorted_xy = camera.undistort_points(img_points)
        cam_data[["undistort_x", "undistort_y"]] = undistorted_xy
        undistorted_data.append(cam_data)

    if not undistorted_data:
        raise ValueError("No valid camera data found for PnP")

    all_undistorted = pd.concat(undistorted_data)
    grouped = all_undistorted.groupby(["port", "sync_index"])

    poses = {}
    success_count = 0
    failure_count = 0

    start_time = time.time()
    K_perfect = np.identity(3)
    D_perfect = np.zeros(5)

    for (port, sync_index), group in grouped:
        if len(group) < min_points:
            failure_count += 1
            continue

        # Use pre-undistorted points
        img_points = group[["undistort_x", "undistort_y"]].to_numpy(dtype=np.float32)
        obj_points = group[["obj_loc_x", "obj_loc_y"]].to_numpy()
        obj_points = np.hstack([obj_points, np.zeros((len(obj_points), 1))]).astype(np.float32)

        # PnP with configurable flags
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, cameraMatrix=K_perfect, distCoeffs=D_perfect, flags=pnp_flags
        )

        if not success:
            success, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, cameraMatrix=K_perfect, distCoeffs=D_perfect, flags=fallback_flags
            )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()

            # Compute reprojection error in normalized space
            projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K_perfect, D_perfect)
            rmse = np.sqrt(np.mean(np.sum((img_points - projected.reshape(-1, 2)) ** 2, axis=1)))

            poses[(port, sync_index)] = (R, t, rmse)
            success_count += 1
        else:
            failure_count += 1

    elapsed = time.time() - start_time
    logger.info(
        f"PnP complete: {success_count} successes, {failure_count} failures "
        f"in {elapsed:.2f}s ({elapsed / max(success_count, 1) * 1000:.2f}ms avg)"
    )
    return poses


def reject_outliers(
    relative_poses: dict[tuple[tuple[int, int], int], StereoPair],
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    rotation_threshold_multiplier: float | None = None,
    translation_threshold_multiplier: float | None = None,
) -> dict[tuple[int, int], list[StereoPair]]:
    """
    Apply IQR-based outlier rejection to relative poses for each camera pair.

    Args:
        relative_poses: dict of relative poses per sync index
        threshold: IQR multiplier for outlier detection (default: 1.5)
        rotation_threshold_multiplier: Optional separate IQR multiplier for rotation outliers
        translation_threshold_multiplier: Optional separate IQR multiplier for translation outliers

    Returns:
        dict mapping pair -> list of StereoPair that passed outlier rejection
    """
    logger.info(f"Applying outlier rejection (threshold={threshold})...")

    # Use separate multipliers if provided, otherwise use the main threshold
    rot_multiplier = rotation_threshold_multiplier if rotation_threshold_multiplier is not None else threshold
    trans_multiplier = translation_threshold_multiplier if translation_threshold_multiplier is not None else threshold

    # Group poses by pair
    poses_by_pair: dict[tuple[int, int], list[StereoPair]] = {}
    for (pair, _sync_index), stereo_pair in relative_poses.items():
        poses_by_pair.setdefault(pair, []).append(stereo_pair)

    filtered_poses = {}
    for pair, stereo_pairs in poses_by_pair.items():
        # Filter NaN poses
        valid_pairs = [
            sp for sp in stereo_pairs if not (np.any(np.isnan(sp.rotation)) or np.any(np.isnan(sp.translation)))
        ]

        if len(valid_pairs) < 5:
            logger.warning(f"Pair {pair} has only {len(valid_pairs)} samples, skipping outlier rejection")
            filtered_poses[pair] = valid_pairs
            continue

        # Extract arrays for statistical analysis
        quats = []
        t_mags = []
        for sp in valid_pairs:
            quat = Rotation.from_matrix(sp.rotation).as_quat()  # (x, y, z, w)
            quats.append(np.roll(quat, 1))  # Convert to (w, x, y, z)
            t_mags.append(np.linalg.norm(sp.translation))

        quats = np.array(quats)
        t_mags = np.array(t_mags)

        # Translation magnitude IQR filter
        t_q1, t_q3 = np.percentile(t_mags, [25, 75])
        t_iqr = t_q3 - t_q1
        t_lower, t_upper = t_q1 - trans_multiplier * t_iqr, t_q3 + trans_multiplier * t_iqr

        # Rotation angle IQR filter (angle from median quaternion)
        median_quat = quaternion_average(quats)
        R_median = Rotation.from_quat(np.roll(median_quat, -1)).as_matrix()
        rot_angles = np.array([rotation_error(sp.rotation, R_median) for sp in valid_pairs])

        rot_q1, rot_q3 = np.percentile(rot_angles, [25, 75])
        rot_iqr = rot_q3 - rot_q1
        rot_upper = rot_q3 + rot_multiplier * rot_iqr

        # Apply filters
        filtered = []
        outlier_count = 0
        for i, stereo_pair in enumerate(valid_pairs):
            is_t_outlier = t_mags[i] < t_lower or t_mags[i] > t_upper
            is_rot_outlier = rot_angles[i] > rot_upper

            if not (is_t_outlier or is_rot_outlier):
                filtered.append(stereo_pair)
            else:
                outlier_count += 1

        logger.info(f"Pair {pair}: {outlier_count}/{len(valid_pairs)} outliers rejected")
        filtered_poses[pair] = filtered

    return filtered_poses


def quaternion_average(quaternions: NDArray[np.float64]) -> np.ndarray:
    if len(quaternions) == 0:
        raise ValueError("Cannot average empty quaternion array")
    if len(quaternions) == 1:
        return quaternions[0]

    # Compute eigenvector of covariance matrix for largest eigenvalue
    Q = quaternions.T
    M = Q @ Q.T
    eigenvals, eigenvecs = np.linalg.eigh(M)

    avg_quat = eigenvecs[:, -1]
    if avg_quat[0] < 0:
        avg_quat = -avg_quat

    # Only safeguard we need: handle numerical instability
    norm = np.linalg.norm(avg_quat)
    if norm < 1e-10:
        logger.warning("Quaternion average failed, returning first quaternion")
        return quaternions[0]

    return avg_quat / norm


# NOTE: used for data inspection, could be removed in future
def rotation_error(R1: NDArray[np.float64], R2: NDArray[np.float64]) -> float:
    """
    Compute geodesic rotation error in degrees between two rotation matrices.

    This is the minimal rotation angle needed to align R1 with R2.

    Args:
        R1, R2: (3, 3) rotation matrices

    Returns:
        Error in degrees
    """
    R_rel = R1 @ R2.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)  # Clamp for numerical stability
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)


# NOTE: used for data inspection, could be removed in future
def translation_error(t1: NDArray[np.float64], t2: NDArray[np.float64]) -> dict[str, float]:
    """
    Compute translation errors between two vectors.

    Args:
        t1, t2: (3,) translation vectors

    Returns:
        dict with magnitude_error (%) and direction_error (degrees)
    """
    mag1, mag2 = np.linalg.norm(t1), np.linalg.norm(t2)

    # Handle near-zero magnitudes
    if mag1 < 1e-10 or mag2 < 1e-10:
        magnitude_error = 0.0 if abs(mag1 - mag2) < 1e-10 else float("inf")
        direction_error = 0.0
    else:
        magnitude_error = abs(mag1 - mag2) / mag1 * 100
        dot_product = np.clip(np.dot(t1 / mag1, t2 / mag2), -1.0, 1.0)
        direction_error = np.degrees(np.arccos(dot_product))

    return {"magnitude_delta_pct": magnitude_error, "direction_delta_deg": direction_error}


def compute_relative_poses(
    camera_to_object_poses: dict[tuple[int, int], tuple[NDArray[np.float64], NDArray[np.float64], float]],
    camera_array: CameraArray,
) -> dict[tuple[tuple[int, int], int], StereoPair]:
    """
    Compute relative poses between camera pairs at each sync_index.

    For cameras A and B observing the same object at a sync index:
    T_B_A = T_B_obj @ T_obj_A = T_B_obj @ inv(T_A_obj)

    Returns:
        dict mapping (pair, sync_index) -> StereoPair with relative pose
    """
    logger.info("Computing relative poses between camera pairs...")
    relative_poses: dict[tuple[tuple[int, int], int], StereoPair] = {}

    ports = [p for p, cam in camera_array.cameras.items() if not cam.ignore]
    pairs = [(i, j) for i, j in combinations(ports, 2) if i < j]

    for port_a, port_b in pairs:
        # Find sync indices where both cameras have poses
        sync_a = {s for p, s in camera_to_object_poses.keys() if p == port_a}
        sync_b = {s for p, s in camera_to_object_poses.keys() if p == port_b}
        common_sync = sync_a.intersection(sync_b)

        for sync_index in common_sync:
            R_a, t_a, _ = camera_to_object_poses[(port_a, sync_index)]
            R_b, t_b, _ = camera_to_object_poses[(port_b, sync_index)]

            # Compute relative transformation: T_B_A = T_B_obj @ inv(T_A_obj)
            R_a_inv = R_a.T
            t_a_inv = -R_a_inv @ t_a
            R_rel = R_b @ R_a_inv
            t_rel = R_b @ t_a_inv + t_b

            # Store as StereoPair (error_score will be computed after aggregation)
            pair_key = (port_a, port_b)
            relative_poses[(pair_key, sync_index)] = StereoPair(
                primary_port=port_a,
                secondary_port=port_b,
                error_score=None,  # Will be filled after RMSE calculation
                translation=t_rel,
                rotation=R_rel,
            )

    logger.info(f"Computed {len(relative_poses)} relative poses across {len(pairs)} pairs")
    return relative_poses


def aggregate_poses(filtered_poses: dict[tuple[int, int], list[StereoPair]]) -> dict[tuple[int, int], StereoPair]:
    """
    Aggregate per-sync-index poses into a single robust estimate per pair.

    Uses quaternion averaging for rotations and simple averaging for translations
    after outlier rejection.

    Returns:
        dict mapping pair -> aggregated StereoPair
    """
    logger.info("Aggregating poses...")

    aggregated = {}
    for pair, stereo_pairs in filtered_poses.items():
        if not stereo_pairs:
            logger.warning(f"No valid poses for pair {pair} after outlier rejection")
            continue

        if len(stereo_pairs) == 1:
            aggregated[pair] = stereo_pairs[0]
            continue

        # Extract and average rotations (via quaternion space)
        quats = [np.roll(Rotation.from_matrix(sp.rotation).as_quat(), 1) for sp in stereo_pairs]
        avg_quat = quaternion_average(np.array(quats))
        avg_R = Rotation.from_quat(np.roll(avg_quat, -1)).as_matrix()

        # Average translations
        translations = [sp.translation for sp in stereo_pairs]
        avg_t = np.mean(translations, axis=0)

        aggregated[pair] = StereoPair(
            primary_port=pair[0],
            secondary_port=pair[1],
            error_score=None,  # Will be filled by RMSE calculation
            rotation=avg_R,
            translation=avg_t,
        )

    logger.info(f"Aggregated poses for {len(aggregated)} pairs")
    return aggregated


def estimate_pnp_paired_pose_network(
    aggregated_pairs_wo_rmse: dict[tuple[int, int], StereoPair], camera_array: CameraArray, image_points: ImagePoints
) -> PairedPoseNetwork:
    """
    Create a PairedPoseNetwork from aggregated stereo pairs with RMSE scores.

    Calculates the Stereo RMSE for each pair and populates the error_score field.

    Returns:
        PairedPoseNetwork ready for use in camera array initialization
    """
    logger.info("Creating PairedPoseNetwork...")

    # Pre-compute common observations for RMSE calculation
    common_observations = _precompute_common_observations(image_points, camera_array)

    pairs_with_rmse = {}
    for pair, stereo_pair in aggregated_pairs_wo_rmse.items():
        rmse = calculate_stereo_rmse_for_pair(stereo_pair, camera_array, common_observations)
        if rmse is None:
            logger.warning(f"Could not compute RMSE for pair {pair}, skipping")
            continue

        # Create new StereoPair with RMSE populated
        pairs_with_rmse[pair] = StereoPair(
            primary_port=stereo_pair.primary_port,
            secondary_port=stereo_pair.secondary_port,
            error_score=rmse,
            rotation=stereo_pair.rotation,
            translation=stereo_pair.translation,
        )
        logger.info(f"Pair {pair}: RMSE = {rmse:.6f}")

    return PairedPoseNetwork.from_raw_estimates(pairs_with_rmse)


def _precompute_common_observations(
    image_points: ImagePoints, camera_array: CameraArray
) -> dict[tuple[int, int], pd.DataFrame]:
    """
    Pre-compute common observations for all camera pairs to avoid repeated merges.

    Returns:
        dict mapping (port_a, port_b) -> DataFrame with common observations
    """
    df = image_points.df
    ports = [p for p, cam in camera_array.cameras.items() if not cam.ignore]

    common_obs = {}
    for port_a, port_b in combinations(ports, 2):
        data_a = df[df["port"] == port_a]
        data_b = df[df["port"] == port_b]

        # Merge once per pair
        common = pd.merge(data_a, data_b, on=["sync_index", "point_id"], suffixes=("_a", "_b"))
        if len(common) >= DEFAULT_MIN_PNP_POINTS:
            common_obs[(port_a, port_b)] = common

    logger.info(f"Pre-computed common observations for {len(common_obs)} pairs")
    return common_obs


def calculate_stereo_rmse_for_pair(
    stereo_pair: StereoPair, camera_array: CameraArray, common_observations: dict[tuple[int, int], pd.DataFrame]
) -> float | None:
    """
    Calculate Stereo RMSE for a pair using triangulation/reprojection error.

    Mimics cv2.stereoCalibrate's internal error calculation:
    1. Undistort points to normalized plane
    2. Triangulate using the provided pose
    3. Project back to both cameras
    4. Calculate RMS error of residuals

    Returns:
        RMSE value or None if insufficient data
    """
    port_a, port_b = stereo_pair.pair

    # Look up pre-computed common observations
    common = common_observations.get((port_a, port_b))
    if common is None or len(common) < DEFAULT_MIN_PNP_POINTS:
        logger.warning(f"Insufficient common points for RMSE calc on pair {stereo_pair}")
        return None

    cam_a, cam_b = camera_array.cameras[port_a], camera_array.cameras[port_b]

    # Extract and undistort points
    pts_a = common[["img_loc_x_a", "img_loc_y_a"]].to_numpy(dtype=np.float32)
    pts_b = common[["img_loc_x_b", "img_loc_y_b"]].to_numpy(dtype=np.float32)
    norm_a = cam_a.undistort_points(pts_a)
    norm_b = cam_b.undistort_points(pts_b)

    # Triangulate using the stereo pair pose
    P1 = np.eye(3, 4)  # Camera A at origin
    P2 = np.hstack((stereo_pair.rotation, stereo_pair.translation.reshape(3, 1)))  # Camera B relative to A

    points_4d = cv2.triangulatePoints(P1, P2, norm_a.T, norm_b.T)
    points_3d = points_4d[:3] / points_4d[3]

    # Project back to normalized plane and compute residuals
    proj_a, _ = cv2.projectPoints(points_3d.T, np.zeros(3), np.zeros(3), np.eye(3), np.zeros(5))
    proj_b, _ = cv2.projectPoints(
        points_3d.T, cv2.Rodrigues(stereo_pair.rotation)[0], stereo_pair.translation, np.eye(3), np.zeros(5)
    )

    errors = np.vstack([norm_a - proj_a.reshape(-1, 2), norm_b - proj_b.reshape(-1, 2)])
    rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

    return float(rmse)
