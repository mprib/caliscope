"""Post-BA metric anchoring: CaptureVolume.scaled / oriented / grounded.

These transforms are exact arithmetic on an already-solved volume, so no bundle
adjustment runs here. Fixtures are hand-built: cameras with known extrinsics, a
handful of world points, and matching minimal image points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.core.scale_cues import CameraDistance, DepthObservation, SegmentLength

DEFAULT_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def make_camera(cam_id: int, rotation: np.ndarray, center: np.ndarray) -> CameraData:
    """Posed camera whose world-frame center is ``center`` (t = -R @ C)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    translation = -rotation @ center
    return CameraData(
        cam_id=cam_id,
        size=(640, 480),
        matrix=DEFAULT_K.copy(),
        distortions=np.zeros(5),
        rotation=rotation,
        translation=translation,
    )


def make_volume(cameras: dict[int, CameraData], world_rows: list[dict]) -> CaptureVolume:
    """Build a CaptureVolume from posed cameras and explicit world points.

    Each world point is given a matching observation in every posed camera so the
    image-to-object map has real correspondences. Image coordinates are dummies --
    nothing here re-triangulates.
    """
    world_df = pd.DataFrame(world_rows)
    world_df["frame_time"] = np.nan
    world_points = WorldPoints(world_df)

    img_rows = []
    for row in world_rows:
        for cam_id in cameras:
            img_rows.append(
                {
                    "sync_index": row["sync_index"],
                    "cam_id": cam_id,
                    "object_id": row["object_id"],
                    "keypoint_id": row["keypoint_id"],
                    "img_loc_x": 100.0,
                    "img_loc_y": 100.0,
                }
            )
    image_points = ImagePoints(pd.DataFrame(img_rows))

    return CaptureVolume(
        camera_array=CameraArray(cameras),
        image_points=image_points,
        world_points=world_points,
    )


def world_row(sync_index: int, keypoint_id: int, xyz: tuple[float, float, float], object_id: int = 0) -> dict:
    return {
        "sync_index": sync_index,
        "object_id": object_id,
        "keypoint_id": keypoint_id,
        "x_coord": float(xyz[0]),
        "y_coord": float(xyz[1]),
        "z_coord": float(xyz[2]),
    }


def camera_center(volume: CaptureVolume, cam_id: int) -> np.ndarray:
    cam = volume.camera_array.cameras[cam_id]
    assert cam.rotation is not None and cam.translation is not None
    return -cam.rotation.T @ cam.translation


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def test_single_camera_distance_recovers_scale():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([3.0, 0.0, 0.0])),
        2: make_camera(2, np.eye(3), np.array([0.0, 3.0, 0.0])),
    }
    rows = [world_row(0, 10, (1.0, 1.0, 1.0)), world_row(0, 11, (2.0, 0.0, 1.0))]
    volume = make_volume(cameras, rows)

    scaled = volume.scaled(CameraDistance(cam_a=0, cam_b=1, meters=7.5))

    recovered = float(np.linalg.norm(camera_center(scaled, 0) - camera_center(scaled, 1)))
    assert abs(recovered - 7.5) < 1e-10


def test_single_segment_length_recovers_scale():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([3.0, 0.0, 0.0])),
    }
    # Keypoints 10 and 11 are exactly one BA-unit apart in every frame,
    # so the median segment length is 1.0.
    rows = [
        world_row(0, 10, (0.0, 0.0, 0.0)),
        world_row(0, 11, (0.0, 0.0, 1.0)),
        world_row(1, 10, (1.0, 0.0, 0.0)),
        world_row(1, 11, (1.0, 0.0, 1.0)),
    ]
    volume = make_volume(cameras, rows)

    scaled = volume.scaled(SegmentLength(keypoint_id_a=10, keypoint_id_b=11, meters=2.5))

    df = scaled.world_points.df
    p10 = df[(df["sync_index"] == 0) & (df["keypoint_id"] == 10)][["x_coord", "y_coord", "z_coord"]].to_numpy()[0]
    p11 = df[(df["sync_index"] == 0) & (df["keypoint_id"] == 11)][["x_coord", "y_coord", "z_coord"]].to_numpy()[0]
    recovered = float(np.linalg.norm(p10 - p11))
    assert abs(recovered - 2.5) < 1e-10


def test_single_depth_observation_recovers_scale():
    # Camera 0 at origin looking down +Z; a point at BA-depth 2 along its axis.
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([3.0, 0.0, 0.0])),
    }
    rows = [world_row(5, 10, (0.0, 0.0, 2.0)), world_row(5, 11, (1.0, 0.0, 2.0))]
    volume = make_volume(cameras, rows)

    scaled = volume.scaled(DepthObservation(cam_id=0, keypoint_id=10, sync_index=5, depth_m=5.0))

    df = scaled.world_points.df
    p = df[(df["sync_index"] == 5) & (df["keypoint_id"] == 10)][["x_coord", "y_coord", "z_coord"]].to_numpy()[0]
    cam = scaled.camera_array.cameras[0]
    recovered_depth = float((cam.rotation @ p + cam.translation)[2])
    assert abs(recovered_depth - 5.0) < 1e-10


def test_multiple_cues_weighted_between_estimates():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([1.0, 0.0, 0.0])),  # d = 1
        2: make_camera(2, np.eye(3), np.array([0.0, 1.0, 0.0])),  # d = 1
    }
    rows = [world_row(0, 10, (0.5, 0.5, 1.0)), world_row(0, 11, (0.2, 0.3, 1.0))]
    volume = make_volume(cameras, rows)

    # Cue A implies scale 2.0 (tight), cue B implies scale 3.0 (loose).
    scaled = volume.scaled(
        CameraDistance(cam_a=0, cam_b=1, meters=2.0, sigma_m=0.001),
        CameraDistance(cam_a=0, cam_b=2, meters=3.0, sigma_m=1.0),
    )

    recovered = float(np.linalg.norm(camera_center(scaled, 0) - camera_center(scaled, 1)))
    assert 2.0 < recovered < 3.0
    assert abs(recovered - 2.0) < abs(recovered - 3.0)


def test_disagreeing_cues_emit_warning():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([1.0, 0.0, 0.0])),
        2: make_camera(2, np.eye(3), np.array([0.0, 1.0, 0.0])),
    }
    rows = [world_row(0, 10, (0.5, 0.5, 1.0)), world_row(0, 11, (0.2, 0.3, 1.0))]
    volume = make_volume(cameras, rows)

    with pytest.warns(UserWarning):
        volume.scaled(
            CameraDistance(cam_a=0, cam_b=1, meters=2.0, sigma_m=0.01),
            CameraDistance(cam_a=0, cam_b=2, meters=3.0, sigma_m=0.01),
        )


def test_two_camera_volume_scales():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([4.0, 0.0, 0.0])),
    }
    rows = [world_row(0, 10, (1.0, 1.0, 1.0)), world_row(0, 11, (2.0, 1.0, 1.0))]
    volume = make_volume(cameras, rows)

    scaled = volume.scaled(CameraDistance(cam_a=0, cam_b=1, meters=10.0))

    recovered = float(np.linalg.norm(camera_center(scaled, 0) - camera_center(scaled, 1)))
    assert abs(recovered - 10.0) < 1e-10


def test_zero_cues_raises():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([3.0, 0.0, 0.0])),
    }
    rows = [world_row(0, 10, (1.0, 1.0, 1.0)), world_row(0, 11, (2.0, 0.0, 1.0))]
    volume = make_volume(cameras, rows)

    with pytest.raises(ValueError):
        volume.scaled()


def test_missing_camera_cue_raises():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([3.0, 0.0, 0.0])),
    }
    rows = [world_row(0, 10, (1.0, 1.0, 1.0)), world_row(0, 11, (2.0, 0.0, 1.0))]
    volume = make_volume(cameras, rows)

    with pytest.raises(ValueError):
        volume.scaled(CameraDistance(cam_a=0, cam_b=99, meters=5.0))


def test_unresolvable_depth_cues_skipped_survivors_scale():
    # Bulk depth cues: one resolves, one references a keypoint with no world point.
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([3.0, 0.0, 0.0])),
    }
    rows = [world_row(5, 10, (0.0, 0.0, 2.0)), world_row(5, 11, (1.0, 0.0, 2.0))]
    volume = make_volume(cameras, rows)

    with pytest.warns(UserWarning, match="Skipped 1 of 2 depth cues"):
        scaled = volume.scaled(
            DepthObservation(cam_id=0, keypoint_id=10, sync_index=5, depth_m=5.0),  # resolves, depth 2
            DepthObservation(cam_id=0, keypoint_id=99, sync_index=5, depth_m=9.0),  # no world point
        )

    # Only the survivor sets scale: depth 2 -> 5.0 exactly.
    df = scaled.world_points.df
    p = df[(df["sync_index"] == 5) & (df["keypoint_id"] == 10)][["x_coord", "y_coord", "z_coord"]].to_numpy()[0]
    cam = scaled.camera_array.cameras[0]
    assert cam.rotation is not None and cam.translation is not None
    recovered_depth = float((cam.rotation @ p + cam.translation)[2])
    assert abs(recovered_depth - 5.0) < 1e-10


def test_all_depth_cues_unresolvable_raises():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([3.0, 0.0, 0.0])),
    }
    rows = [world_row(5, 10, (0.0, 0.0, 2.0)), world_row(5, 11, (1.0, 0.0, 2.0))]
    volume = make_volume(cameras, rows)

    with pytest.raises(ValueError, match="unresolvable"):
        volume.scaled(
            DepthObservation(cam_id=0, keypoint_id=98, sync_index=5, depth_m=5.0),
            DepthObservation(cam_id=0, keypoint_id=99, sync_index=5, depth_m=9.0),
        )


def test_negative_depth_cue_skipped_camera_distance_still_scales():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([4.0, 0.0, 0.0])),
    }
    # Keypoint 10 sits behind camera 0 (z < 0 in its frame) -- triangulation noise.
    rows = [world_row(0, 10, (0.0, 0.0, -3.0)), world_row(0, 11, (1.0, 0.0, 2.0))]
    volume = make_volume(cameras, rows)

    with pytest.warns(UserWarning, match="non-positive depth"):
        scaled = volume.scaled(
            DepthObservation(cam_id=0, keypoint_id=10, sync_index=0, depth_m=5.0),  # behind camera
            CameraDistance(cam_a=0, cam_b=1, meters=10.0),  # good, sets scale exactly
        )

    recovered = float(np.linalg.norm(camera_center(scaled, 0) - camera_center(scaled, 1)))
    assert abs(recovered - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def _tilted_rig() -> tuple[CaptureVolume, dict[int, np.ndarray], np.ndarray]:
    """A rig whose true vertical is a non-axis direction in the BA frame.

    Returns the volume, the per-camera up dict (up expressed in each camera frame),
    and the true BA-frame vertical. A vertical segment (keypoints 10->11) lies along
    that vertical, so orientation should map it onto +Z.
    """
    true_up = np.array([0.1, 1.0, -0.05])
    true_up = true_up / np.linalg.norm(true_up)

    rotations = {
        0: Rotation.from_euler("xyz", [10, 20, 5], degrees=True).as_matrix(),
        1: Rotation.from_euler("xyz", [-15, 40, 10], degrees=True).as_matrix(),
        2: Rotation.from_euler("xyz", [5, -30, -8], degrees=True).as_matrix(),
    }
    centers = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([2.0, 0.0, 0.0]),
        2: np.array([0.0, 2.0, 0.0]),
    }
    cameras = {cid: make_camera(cid, rotations[cid], centers[cid]) for cid in rotations}

    p_low = np.array([1.0, 2.0, 3.0])
    p_high = p_low + 2.0 * true_up
    rows = [
        world_row(0, 10, tuple(p_low)),
        world_row(0, 11, tuple(p_high)),
        world_row(1, 10, tuple(p_low + 0.3 * true_up)),
        world_row(1, 11, tuple(p_high + 0.3 * true_up)),
    ]
    volume = make_volume(cameras, rows)

    # Up in each camera frame is R_cam @ true_up (world direction -> camera frame).
    up = {cid: rotations[cid] @ true_up for cid in rotations}
    return volume, up, true_up


def test_oriented_levels_known_tilt():
    volume, up, _ = _tilted_rig()

    oriented = volume.oriented(up=up)

    df = oriented.world_points.df
    p10 = df[(df["sync_index"] == 0) & (df["keypoint_id"] == 10)][["x_coord", "y_coord", "z_coord"]].to_numpy()[0]
    p11 = df[(df["sync_index"] == 0) & (df["keypoint_id"] == 11)][["x_coord", "y_coord", "z_coord"]].to_numpy()[0]
    segment = p11 - p10
    # The vertical segment now points straight up +Z with length 2.
    assert np.allclose(segment, [0.0, 0.0, 2.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def test_grounded_puts_floor_at_zero_and_cam0_over_origin():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([5.0, 7.0, 3.0])),
        1: make_camera(1, np.eye(3), np.array([8.0, 7.0, 3.0])),
    }
    rows = [
        world_row(0, 10, (1.0, 1.0, 2.0)),  # lowest z = 2.0
        world_row(0, 11, (1.0, 1.0, 4.0)),
    ]
    volume = make_volume(cameras, rows)

    grounded = volume.grounded("lowest_point")

    assert abs(float(grounded.world_points.df["z_coord"].min())) < 1e-10
    c0 = camera_center(grounded, 0)
    assert abs(c0[0]) < 1e-10
    assert abs(c0[1]) < 1e-10


def test_grounded_rejects_unknown_mode():
    cameras = {
        0: make_camera(0, np.eye(3), np.array([0.0, 0.0, 0.0])),
        1: make_camera(1, np.eye(3), np.array([3.0, 0.0, 0.0])),
    }
    rows = [world_row(0, 10, (1.0, 1.0, 1.0)), world_row(0, 11, (2.0, 0.0, 1.0))]
    volume = make_volume(cameras, rows)

    with pytest.raises(ValueError):
        volume.grounded("centroid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chain order invariance
# ---------------------------------------------------------------------------


def test_chain_order_invariance_scale_and_orient():
    volume, up, _ = _tilted_rig()
    cue = CameraDistance(cam_a=0, cam_b=1, meters=6.0)

    a = volume.scaled(cue).oriented(up=up)
    b = volume.oriented(up=up).scaled(cue)

    assert np.allclose(a.world_points.points, b.world_points.points, atol=1e-12)
    for cam_id in a.camera_array.cameras:
        assert np.allclose(camera_center(a, cam_id), camera_center(b, cam_id), atol=1e-12)


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    test_single_camera_distance_recovers_scale()
    test_single_segment_length_recovers_scale()
    test_single_depth_observation_recovers_scale()
    test_multiple_cues_weighted_between_estimates()
    test_two_camera_volume_scales()
    test_oriented_levels_known_tilt()
    test_grounded_puts_floor_at_zero_and_cam0_over_origin()
    test_chain_order_invariance_scale_and_orient()
    print("anchoring debug run complete")
