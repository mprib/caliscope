"""Factory functions for common synthetic calibration scenes."""

import numpy as np

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.aruco_marker import ArucoMarkerSet
from caliscope.core.constraints import ConstraintSet
from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer, MACHINE_VISION
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
