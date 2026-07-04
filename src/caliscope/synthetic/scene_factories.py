"""Factory functions for common synthetic calibration scenes."""

import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.aruco_marker import ArucoMarkerSet
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import ImagePoints
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer, MACHINE_VISION
from caliscope.synthetic.outliers import OutlierConfig, inject_outliers
from caliscope.synthetic.target_factories import double_sided_charuco_board
from caliscope.synthetic.se3_pose import SE3Pose
from caliscope.synthetic.synthetic_scene import SceneObject, SyntheticScene
from caliscope.synthetic.trajectory import Trajectory


def default_ring_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """Standard 4-camera ring with full orbital trajectory.

    Configuration:
    - 4 cameras in ring, radius=2m, height=0.5m
    - 5x7 planar grid, spacing=0.05m
    - 20 frames, orbital radius=0.2m, full 360 degree orbit
    """
    camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
    calibration_object = CalibrationObject.planar_grid(rows=5, cols=7, spacing=0.05)
    trajectory = Trajectory.orbital(
        n_frames=20,
        radius=0.2,
        arc_extent_deg=360.0,
        tumble_rate=1.0,
    )

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def sparse_coverage_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """4 cameras, partial arc (cameras don't all see same frames)."""
    camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
    calibration_object = CalibrationObject.planar_grid(rows=5, cols=7, spacing=0.05)
    trajectory = Trajectory.orbital(
        n_frames=20,
        radius=0.4,
        arc_extent_deg=180.0,
        tumble_rate=0.5,
    )

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def quick_test_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """Minimal scene for fast tests (5 frames, small grid)."""
    camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
    calibration_object = CalibrationObject.planar_grid(rows=3, cols=4, spacing=0.05)
    trajectory = Trajectory.orbital(
        n_frames=5,
        radius=0.2,
        arc_extent_deg=180.0,
        tumble_rate=0.5,
    )

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def machine_vision_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """4-camera ring with MACHINE_VISION lens profile (KITTI-class barrel distortion)."""
    camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5, lens=MACHINE_VISION).build()
    calibration_object = CalibrationObject.planar_grid(rows=5, cols=7, spacing=0.05)
    trajectory = Trajectory.orbital(
        n_frames=20,
        radius=0.2,
        arc_extent_deg=360.0,
        tumble_rate=1.0,
    )

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def charuco_target_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """4-camera ring with a double-sided charuco board (visible from both sides)."""
    camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
    calibration_object = double_sided_charuco_board(rows=5, cols=7, square_size=0.05)
    trajectory = Trajectory.orbital(
        n_frames=20,
        radius=0.2,
        arc_extent_deg=360.0,
        tumble_rate=1.0,
    )

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def _look_at_camera(position: np.ndarray, target: np.ndarray, cam_id: int) -> CameraData:
    """Create a camera at `position` looking toward `target`."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)

    # Camera +Z = forward (OpenCV convention: camera looks along its +Z)
    # Choose world +Y as provisional up
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    # R maps world → camera: rows are camera axes in world coordinates
    R = np.array([right, down, forward], dtype=np.float64)
    t = (-R @ position).astype(np.float64)

    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)

    return CameraData(cam_id=cam_id, size=(640, 480), matrix=K, distortions=D, rotation=R, translation=t)


def cheirality_demo_scene(
    pixel_noise_sigma: float = 0.0,
    random_seed: int = 42,
) -> SyntheticScene:
    """Two cameras: one facing the object, one facing away.

    Demonstrates the cheirality check — the backward camera produces
    zero observations even though it's close to the object.
    """
    obj = CalibrationObject.planar_grid(rows=3, cols=4, spacing=0.05)
    traj = Trajectory.stationary(n_frames=1)

    cam_toward = _look_at_camera(np.array([0.0, 0.5, 1.5]), np.array([0.0, 0.0, 0.0]), cam_id=0)

    # Camera behind the object, pointing away
    R_away = np.eye(3, dtype=np.float64)
    t_away = np.array([0.0, 0.0, -1.5], dtype=np.float64)
    cam_away = CameraData(
        cam_id=1,
        size=(640, 480),
        matrix=cam_toward.matrix,
        distortions=cam_toward.distortions,
        rotation=R_away,
        translation=t_away,
    )

    cameras = CameraArray(cameras={0: cam_toward, 1: cam_away})
    return SyntheticScene.single(
        camera_array=cameras,
        calibration_object=obj,
        trajectory=traj,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def visibility_culling_scene(
    pixel_noise_sigma: float = 0.0,
    random_seed: int = 42,
) -> SyntheticScene:
    """Two cameras on opposite sides of a single-sided board.

    The board has face_normal +Z. The camera above (+Z side) sees
    all points; the camera below (-Z side) sees nothing.
    """
    obj = CalibrationObject(
        points=np.array(
            [[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]],
            dtype=np.float64,
        ),
        keypoint_ids=np.arange(4, dtype=np.int64),
        face_normal=np.array([0.0, 0.0, 1.0]),
    )
    traj = Trajectory.stationary(n_frames=1)

    cam_above = _look_at_camera(np.array([0.05, 0.05, 1.5]), np.array([0.05, 0.05, 0.0]), cam_id=0)
    cam_below = _look_at_camera(np.array([0.05, 0.05, -1.5]), np.array([0.05, 0.05, 0.0]), cam_id=1)

    cameras = CameraArray(cameras={0: cam_above, 1: cam_below})
    return SyntheticScene.single(
        camera_array=cameras,
        calibration_object=obj,
        trajectory=traj,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def aruco_multi_object_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """One mobile marker orbiting, one static marker offset.

    4-camera ring sees both. Exercises multi-object scenes
    with mixed static/dynamic trajectories.
    """
    import cv2
    from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet

    camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()

    markers = {
        0: ArucoMarker(0, 0.1),
        1: ArucoMarker(1, 0.1, static=True),
    }
    marker_set = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)

    mobile_traj = Trajectory.orbital(n_frames=20, radius=0.5)
    static_pose = SE3Pose.from_axis_angle(
        axis=np.array([0.0, 0.0, 1.0]),
        angle_rad=0.0,
        translation=np.array([0.3, -0.2, 0.0]),
    )
    static_traj = Trajectory.stationary(n_frames=20, pose=static_pose)

    trajectories = {0: mobile_traj, 1: static_traj}
    scene, _constraints = aruco_scene(
        marker_set=marker_set,
        trajectories=trajectories,
        camera_array=camera_array,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )
    return scene


def aruco_scene(
    marker_set: ArucoMarkerSet,
    trajectories: dict[int, Trajectory],
    camera_array: CameraArray,
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
    single_sided: bool = False,
) -> tuple[SyntheticScene, ConstraintSet]:
    """Build a SyntheticScene and matching ConstraintSet from an ArucoMarkerSet.

    Markers without an entry in `trajectories` are skipped.
    """
    objects = []
    for marker_id, marker in marker_set.markers.items():
        if marker_id not in trajectories:
            continue
        cal_obj = CalibrationObject(
            points=np.asarray(marker.corners, dtype=np.float64),
            keypoint_ids=np.arange(len(marker.corners), dtype=np.int64),
            face_normal=np.array([0.0, 0.0, 1.0]) if single_sided else None,
        )
        objects.append(
            SceneObject(
                object_id=marker_id,
                calibration_object=cal_obj,
                trajectory=trajectories[marker_id],
                static=marker.static,
            )
        )

    scene = SyntheticScene(
        camera_array=camera_array,
        objects=tuple(objects),
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )
    constraints = ConstraintSet.from_marker_set(marker_set)
    return scene, constraints


def chain_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """6 cameras in a line with neighbors-only FOV overlap.

    WEBCAM lens (~69 deg HFOV) at distance=1.5m gives half-width ~1.03m.
    Spacing=1.5m means adjacent cameras overlap (~0.56m) but next-nearest
    cameras do not (2*1.03 - 3.0 < 0). Double-sided board so orientation
    never starves an end camera. Height=0.5m avoids coplanar degeneracy.
    """
    camera_array = CameraSynthesizer().add_line(n=6, spacing=1.5, distance=1.5, height=0.5).build()
    calibration_object = double_sided_charuco_board(rows=5, cols=7, square_size=0.05)
    trajectory = Trajectory.linear(
        n_frames=90,
        start=np.array([-4.5, 0.0, 0.0]),
        end=np.array([4.5, 0.0, 0.0]),
        tumble_rate=0.3,
        origin_frame=45,
    )

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def narrow_baseline_scene(
    spacing: float = 0.1,
    distance: float = 5.0,
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """Two cameras with configurable baseline viewing a distant board.

    Default: 0.1m baseline, board at 5m. Use spacing=2.0 for wide control.
    Height=0.5m avoids coplanar degeneracy (board at z=0 projecting as a
    horizontal line when cameras are also at z=0).
    """
    camera_array = CameraSynthesizer().add_line(n=2, spacing=spacing, distance=distance, height=0.5).build()
    calibration_object = double_sided_charuco_board(rows=5, cols=7, square_size=0.1)
    trajectory = Trajectory.linear(
        n_frames=20,
        start=np.array([-0.3, 0.0, 0.0]),
        end=np.array([0.3, 0.0, 0.0]),
        tumble_rate=1.0,
    )

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def outlier_scene(
    outlier_fraction: float = 0.05,
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> tuple[SyntheticScene, ImagePoints, NDArray[np.int64]]:
    """Default ring scene with injected outliers.

    Returns (scene, corrupted_image_points, corrupted_indices).
    """
    scene = default_ring_scene(
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )
    config = OutlierConfig(
        fraction=outlier_fraction,
        magnitude_range=(10.0, 50.0),
        random_seed=random_seed,
    )
    corrupted, indices = inject_outliers(scene.image_points_noisy, config)
    return scene, corrupted, indices


def intrinsic_perturbation_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """4-camera ring with a large charuco board on a near-to-far diagonal trajectory.

    Depth variation and image-periphery coverage make focal length jointly
    observable with extrinsics, for testing joint intrinsic+extrinsic bundle
    adjustment. Trajectory suitability is checked against three premise floors
    (periphery coverage, per-camera depth range, per-camera observation spread)
    so a future trajectory edit that breaks the scene's usefulness fails loudly
    instead of silently.
    """
    camera_array = CameraSynthesizer().add_ring(n=4, radius=1.2, height=0.3).build()
    calibration_object = double_sided_charuco_board(rows=7, cols=10, square_size=0.04)
    trajectory = Trajectory.linear(
        n_frames=40,
        start=np.array([0.7, -0.7, -0.3]),
        end=np.array([-0.7, 0.7, 3.5]),
        tumble_rate=2.0,
        origin_frame=0,
    )

    scene = SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )

    _check_intrinsic_perturbation_premises(scene)
    return scene


def _check_intrinsic_perturbation_premises(scene: SyntheticScene) -> None:
    """Verify the scene has enough depth/periphery variation for focal length to be observable.

    Raises ValueError with details if any floor is violated, so a change to the
    trajectory parameters that erodes scene suitability is caught immediately.
    """
    image_df = scene.image_points_noisy.df
    world_df = scene.world_points.df
    merged = image_df.merge(world_df, on=["sync_index", "object_id", "keypoint_id"], how="inner")

    if merged.empty:
        raise ValueError("intrinsic_perturbation_scene: no observations generated")

    total_peripheral = 0
    total_observations = 0

    for cam_id, camera in scene.camera_array.cameras.items():
        cam_obs = merged[merged["cam_id"] == cam_id]
        if cam_obs.empty:
            raise ValueError(f"intrinsic_perturbation_scene: cam {cam_id} has no observations")

        w, h = camera.size
        half_diag = 0.5 * np.sqrt(w**2 + h**2)
        dx = cam_obs["img_loc_x"].to_numpy() - w / 2.0
        dy = cam_obs["img_loc_y"].to_numpy() - h / 2.0
        dist_from_center = np.sqrt(dx**2 + dy**2)
        peripheral = dist_from_center > 0.6 * half_diag
        total_peripheral += int(peripheral.sum())
        total_observations += len(cam_obs)

        assert camera.rotation is not None and camera.translation is not None
        world_pts = cam_obs[["x_coord", "y_coord", "z_coord"]].to_numpy()
        depths = ((camera.rotation @ world_pts.T).T + camera.translation)[:, 2]
        depth_ratio = depths.max() / depths.min()
        if depth_ratio < 2.0:
            raise ValueError(
                f"intrinsic_perturbation_scene: cam {cam_id} depth ratio {depth_ratio:.2f} < 2.0 floor "
                f"(min={depths.min():.2f}m, max={depths.max():.2f}m)"
            )

        x_span = cam_obs["img_loc_x"].max() - cam_obs["img_loc_x"].min()
        y_span = cam_obs["img_loc_y"].max() - cam_obs["img_loc_y"].min()
        if x_span < 0.25 * w:
            raise ValueError(
                f"intrinsic_perturbation_scene: cam {cam_id} x-spread {x_span:.1f}px < 25% of width ({0.25 * w:.1f}px)"
            )
        if y_span < 0.25 * h:
            raise ValueError(
                f"intrinsic_perturbation_scene: cam {cam_id} y-spread {y_span:.1f}px < 25% of height ({0.25 * h:.1f}px)"
            )

    peripheral_fraction = total_peripheral / total_observations
    if peripheral_fraction < 0.20:
        raise ValueError(f"intrinsic_perturbation_scene: periphery coverage {peripheral_fraction:.1%} < 20% floor")


def large_ring_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """15-camera ring for stress testing."""
    camera_array = CameraSynthesizer().add_ring(n=15, radius=2.5, height=0.5).build()
    calibration_object = CalibrationObject.planar_grid(rows=5, cols=7, spacing=0.05)
    trajectory = Trajectory.orbital(
        n_frames=200,
        radius=0.2,
        arc_extent_deg=360.0,
        tumble_rate=1.0,
    )

    return SyntheticScene.single(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )
