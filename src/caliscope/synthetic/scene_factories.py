"""Factory functions for common synthetic calibration scenes."""

import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet, DistanceLink, MirrorPair
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import ImagePoints
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer, MACHINE_VISION
from caliscope.synthetic.outliers import OutlierConfig, inject_outliers
from caliscope.synthetic.target_factories import box_target, double_sided_charuco_board
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


def box_target_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """4-camera ring with a non-planar box target (visible from all sides).

    Same ring arrangement, frame count, and trajectory as default_ring_scene,
    but the target is a 0.4m box (8 corners + 6 face centers). Its z spread
    makes the bootstrap classify each view as non-planar and solve it with
    SQPNP.
    """
    camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
    calibration_object = box_target(width=0.4, height=0.4, depth=0.4)
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


def wand_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """Product-workflow scene: wand with 2 linked ArUcos + 2 static wall markers.

    4-camera ring, WEBCAM lens, diagonal trajectory with depth variation.
    The wand is a rigid body with markers 0 and 1 separated by 30cm.
    Static markers 2 and 3 sit on opposite walls. Constraints include
    6 intra-marker distances per marker + 4 cross-marker distances for
    the wand link.

    Use wand_scene_with_constraints() to get the scene and ConstraintSet
    together.
    """
    scene, _constraints = wand_scene_with_constraints(
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
        include_static=True,
    )
    return scene


def wand_scene_with_constraints(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
    include_static: bool = True,
) -> tuple[SyntheticScene, ConstraintSet]:
    """Product-workflow scene with constraints for joint BA experiments.

    Two 30cm ArUco markers on a rigid wand (50cm separation), diagonal
    trajectory with depth variation. Optionally includes two static wall
    markers. Static markers have a known triangulation issue with noisy
    4-corner observations — set include_static=False to exclude them.

    The wand markers are composed by a pure X-translation offset (identity
    rotation, see `wand_offset` below) applied to every base pose, so the 16
    cross-marker corner distances and the center distance are frame-invariant
    and are declared here as exact `DistanceLink`s computed from marker-local
    geometry, rather than a single uniform distance asserted across all four
    corner pairs (which only happened to be exact for this pure-translation
    case).
    """
    import cv2
    from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet, DistanceLink

    camera_array = CameraSynthesizer().add_ring(n=4, radius=1.2, height=0.3).build()

    wand_separation = 0.50
    markers: dict[int, ArucoMarker] = {
        0: ArucoMarker(0, 0.30),
        1: ArucoMarker(1, 0.30),
    }
    if include_static:
        markers[2] = ArucoMarker(2, 0.30, static=True)
        markers[3] = ArucoMarker(3, 0.30, static=True)
        markers[4] = ArucoMarker(4, 0.30, static=True)
        markers[5] = ArucoMarker(5, 0.30, static=True)

    # All 16 corner-to-corner distances plus the center-to-center distance,
    # computed from marker-local corner geometry offset by the wand's pure
    # X-translation — exact, not measured or hardcoded.
    corners_0 = markers[0].corners
    corners_1 = corners_0 + np.array([wand_separation, 0.0, 0.0])
    links: list[DistanceLink] = [
        DistanceLink(
            marker_a=0,
            corner_a=i,
            marker_b=1,
            corner_b=j,
            distance_m=float(np.linalg.norm(corners_0[i] - corners_1[j])),
        )
        for i in range(4)
        for j in range(4)
    ]
    links.append(
        DistanceLink(
            marker_a=0,
            marker_b=1,
            distance_m=float(np.linalg.norm(corners_0.mean(axis=0) - corners_1.mean(axis=0))),
        )
    )

    static_marker_ids = (2, 3, 4, 5)
    static_layout: dict[int, tuple[float, NDArray[np.float64]]] = {}
    if include_static:
        static_radius = 1.0
        for i, marker_id in enumerate(static_marker_ids):
            angle = np.radians(45 + 90 * i)
            pos = np.array([static_radius * np.cos(angle), static_radius * np.sin(angle), 0.0])
            static_layout[marker_id] = (angle, pos)

        # A marker's center is its own local origin, so rotating it about that
        # origin doesn't move the center — the static markers' centers sit
        # exactly at their placement positions. One static-static center link
        # across the ring diagonal (markers 2, 4) covers the tape-measure
        # scale-anchor case with a long, exactly-known lever arm.
        links.append(
            DistanceLink(
                marker_a=2,
                marker_b=4,
                distance_m=float(np.linalg.norm(static_layout[2][1] - static_layout[4][1])),
            )
        )

    marker_set = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers, links=tuple(links))

    wand_base_trajectory = Trajectory.linear(
        n_frames=40,
        start=np.array([0.7, -0.7, -0.3]),
        end=np.array([-0.7, 0.7, 3.5]),
        tumble_rate=2.0,
        origin_frame=0,
    )

    wand_offset = SE3Pose.from_axis_angle(
        axis=np.array([0.0, 0.0, 1.0]),
        angle_rad=0.0,
        translation=np.array([wand_separation, 0.0, 0.0]),
    )
    offset_poses = tuple(wand_offset.compose(p) for p in wand_base_trajectory.poses)
    wand_tip_trajectory = Trajectory(
        poses=offset_poses,
        origin_frame=wand_base_trajectory.origin_frame,
    )

    trajectories: dict[int, Trajectory] = {
        0: wand_base_trajectory,
        1: wand_tip_trajectory,
    }
    if include_static:
        # 4 static markers on the floor between cameras, facing upward
        n_frames = 40
        for marker_id, (angle, pos) in static_layout.items():
            # Markers lie flat on the floor (no rotation = face up along +Z)
            static_pose = SE3Pose.from_axis_angle(
                axis=np.array([0.0, 0.0, 1.0]),
                angle_rad=angle,
                translation=pos,
            )
            trajectories[marker_id] = Trajectory.stationary(n_frames=n_frames, pose=static_pose)

    return aruco_scene(
        marker_set=marker_set,
        trajectories=trajectories,
        camera_array=camera_array,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


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


# --- Mirror-pair scenes -----------------------------------------------------
#
# A mirror pair is two same-size ArUco markers printed on opposite faces of a
# rigid board. These factories build the two faces as two single-sided
# SceneObjects whose corners coincide (zero thickness) or sit `thickness_m`
# apart (nonzero) per the anchor-derived corner mapping, so the compiled
# ConstraintSet's remaps/thickness constraints exercise the real pipeline.


def _mirror_flip_pose(marker_a: ArucoMarker, marker_b: ArucoMarker, pair: MirrorPair) -> SE3Pose:
    """Board-local rigid flip F with F.apply(C_B[b]) == C_A[a] + (0,0,-thickness).

    Recovered by a Kabsch fit of marker B's corners (in mapping order) onto
    marker A's corners. Both corner sets are centered squares, so the fit is a
    pure proper rotation (a 180-degree turn about an in-plane axis) that also
    flips the face normal to the opposite side. The translation carries the
    board thickness along marker A's local normal (-Z).
    """
    c_a = marker_a.corners
    c_b = marker_b.corners
    b_src = np.array([c_b[b] for _, b in pair.corner_mapping])
    a_tgt = np.array([c_a[a] for a, _ in pair.corner_mapping])
    h = b_src.T @ a_tgt
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return SE3Pose(rotation=rotation, translation=np.array([0.0, 0.0, -pair.thickness_m]))


def _facing_camera_pair(distance_m: float) -> CameraArray:
    """Two cameras on the +Z and -Z axes, each looking at the origin."""
    cam_plus = _look_at_camera(np.array([0.0, 0.0, distance_m]), np.zeros(3), cam_id=0)
    cam_minus = _look_at_camera(np.array([0.0, 0.0, -distance_m]), np.zeros(3), cam_id=1)
    return CameraArray(cameras={0: cam_plus, 1: cam_minus})


def _tilt_sweep_trajectory(n_frames: int, max_tilt_deg: float = 35.0, xy_amplitude: float = 0.15) -> Trajectory:
    """Board centered near the origin, sweeping tilt about both in-plane axes.

    The out-of-plane tilt is essential: two facing cameras viewing a single
    planar marker sit on the classic two-fold planar-pose ambiguity when the
    board is fronto-parallel. Sweeping the tilt (never holding it flat) breaks
    the branch symmetry. A small XY translation spreads the corners across the
    image.
    """
    poses = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        tilt_x = np.radians(max_tilt_deg) * np.sin(2 * np.pi * t)
        tilt_y = np.radians(max_tilt_deg) * np.cos(2 * np.pi * t * 1.3)
        rx = np.array(
            [[1, 0, 0], [0, np.cos(tilt_x), -np.sin(tilt_x)], [0, np.sin(tilt_x), np.cos(tilt_x)]],
            dtype=np.float64,
        )
        ry = np.array(
            [[np.cos(tilt_y), 0, np.sin(tilt_y)], [0, 1, 0], [-np.sin(tilt_y), 0, np.cos(tilt_y)]],
            dtype=np.float64,
        )
        pos = np.array([xy_amplitude * np.sin(2 * np.pi * t), xy_amplitude * np.cos(2 * np.pi * t), 0.0])
        poses.append(SE3Pose(rotation=rx @ ry, translation=pos))
    return Trajectory(poses=tuple(poses), origin_frame=0)


def _spin_tumble_trajectory(n_frames: int, tilt_deg: float = 20.0) -> Trajectory:
    """Board spins a full turn about world Y with a fixed X-tilt.

    The face normal precesses through the XZ plane, pointing at each ring
    camera in turn, so each face is co-observed by several cameras across the
    trajectory. That chains an otherwise two-component pose graph (one per
    opaque face) into a single connected graph — the prerequisite for the
    thickness constraints to pull the faces together during bundle adjustment.
    The fixed tilt keeps the board off exact edge-on to the ring plane.
    """
    poses = []
    for i in range(n_frames):
        t = i / n_frames  # periodic: no duplicated endpoint
        spin = 2 * np.pi * t
        tx = np.radians(tilt_deg)
        ry = np.array(
            [[np.cos(spin), 0, np.sin(spin)], [0, 1, 0], [-np.sin(spin), 0, np.cos(spin)]],
            dtype=np.float64,
        )
        rx = np.array(
            [[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]],
            dtype=np.float64,
        )
        poses.append(SE3Pose(rotation=ry @ rx, translation=np.zeros(3)))
    return Trajectory(poses=tuple(poses), origin_frame=0)


def _face_object(marker: ArucoMarker) -> CalibrationObject:
    """Single-sided calibration object for one marker face (normal +Z local)."""
    return CalibrationObject(
        points=np.asarray(marker.corners, dtype=np.float64),
        keypoint_ids=np.arange(4, dtype=np.int64),
        face_normal=np.array([0.0, 0.0, 1.0]),
    )


def mirror_pair_two_camera_scene(
    anchor_corner_a: int = 0,
    anchor_corner_b: int = 2,
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> tuple[SyntheticScene, ArucoMarkerSet]:
    """Zero-thickness mirror board, two cameras facing each other.

    One marker per face of a thin board; cam 0 sees face A (marker 0), cam 1
    sees face B (marker 1). After the compiled ConstraintSet remaps face B's
    observations onto marker A's identity, both cameras contribute to the same
    triangulated world points and connect in the pose graph. The board sweeps
    through a tilt range (design decision 11) to avoid the planar-pose
    ambiguity two facing cameras would otherwise sit on.
    """
    import cv2

    size = 0.165
    marker_a = ArucoMarker(0, size)
    marker_b = ArucoMarker(1, size)
    pair = MirrorPair(
        marker_a=0,
        marker_b=1,
        anchor_corner_a=anchor_corner_a,
        anchor_corner_b=anchor_corner_b,
        thickness_m=0.0,
    )
    marker_set = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_50,
        markers={0: marker_a, 1: marker_b},
        mirror_pairs=(pair,),
    )

    board_trajectory = _tilt_sweep_trajectory(n_frames=24)
    flip = _mirror_flip_pose(marker_a, marker_b, pair)
    face_b_trajectory = Trajectory(
        poses=tuple(flip.compose(p) for p in board_trajectory.poses),
        origin_frame=board_trajectory.origin_frame,
    )

    scene = SyntheticScene(
        camera_array=_facing_camera_pair(distance_m=1.2),
        objects=(
            SceneObject(object_id=0, calibration_object=_face_object(marker_a), trajectory=board_trajectory),
            SceneObject(object_id=1, calibration_object=_face_object(marker_b), trajectory=face_b_trajectory),
        ),
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )
    return scene, marker_set


def mirror_pair_ring_scene(
    thickness_m: float = 0.005,
    anchor_corner_a: int = 0,
    anchor_corner_b: int = 2,
    with_anchored_marker: bool = False,
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> tuple[SyntheticScene, ArucoMarkerSet]:
    """Mirror board in a 4-camera ring, board spinning through a full turn.

    Defaults to a 5mm-thick board: each face keeps its own identity and the
    compiled ConstraintSet carries four thickness DistanceConstraints between
    corresponding corners. The spin lets each face sweep across multiple ring
    cameras so the pose graph connects across faces.

    with_anchored_marker adds a third static marker (id 2) and a center
    DistanceLink from marker A (id 0) to it, exercising mirror pairs and an
    explicit link to a third target in the same compile.
    """
    import cv2

    marker_a = ArucoMarker(0, 0.165)
    marker_b = ArucoMarker(1, 0.165)
    markers: dict[int, ArucoMarker] = {0: marker_a, 1: marker_b}
    pair = MirrorPair(
        marker_a=0,
        marker_b=1,
        anchor_corner_a=anchor_corner_a,
        anchor_corner_b=anchor_corner_b,
        thickness_m=thickness_m,
    )

    board_trajectory = _spin_tumble_trajectory(n_frames=40)
    flip = _mirror_flip_pose(marker_a, marker_b, pair)
    face_b_trajectory = Trajectory(
        poses=tuple(flip.compose(p) for p in board_trajectory.poses),
        origin_frame=board_trajectory.origin_frame,
    )

    objects = [
        SceneObject(object_id=0, calibration_object=_face_object(marker_a), trajectory=board_trajectory),
        SceneObject(object_id=1, calibration_object=_face_object(marker_b), trajectory=face_b_trajectory),
    ]
    links: tuple[DistanceLink, ...] = ()

    if with_anchored_marker:
        marker_c = ArucoMarker(2, 0.165, static=False)
        markers[2] = marker_c
        # Third marker rigidly co-planar with face A, offset in the board's
        # local frame, so it is co-observed with face A on every frame and its
        # link to marker A reliably fires. The offset lives in the trajectory,
        # not the local corners, matching what the ArUco tracker emits.
        neighbor_offset = 0.4
        offset_pose = SE3Pose(rotation=np.eye(3), translation=np.array([neighbor_offset, 0.0, 0.0]))
        neighbor_trajectory = Trajectory(
            poses=tuple(offset_pose.compose(p) for p in board_trajectory.poses),
            origin_frame=board_trajectory.origin_frame,
        )
        objects.append(
            SceneObject(object_id=2, calibration_object=_face_object(marker_c), trajectory=neighbor_trajectory)
        )
        # Rigid in-plane offset, so the center-to-center distance is exactly the
        # offset magnitude and frame-invariant.
        links = (DistanceLink(marker_a=0, marker_b=2, distance_m=neighbor_offset),)

    marker_set = ArucoMarkerSet(
        dictionary=cv2.aruco.DICT_4X4_50,
        markers=markers,
        links=links,
        mirror_pairs=(pair,),
    )

    scene = SyntheticScene(
        camera_array=CameraSynthesizer().add_ring(n=4, radius=1.2, height=0.0).build(),
        objects=tuple(objects),
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )
    return scene, marker_set
