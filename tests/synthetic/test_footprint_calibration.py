"""Markerless calibration from each camera's primary track, with no extrinsics."""

from __future__ import annotations

from caliscope.core.calibrate_extrinsics import calibrate_extrinsics
from caliscope.core.point_data import ImagePoints
from tests.synthetic.tracked_scene import build_scene, ring_cameras, standing, to_tracked, walking

N_FRAMES = 30
SUBJECT = 0


def test_primary_tracks_follow_the_walking_subject_and_calibration_poses_every_camera():
    cameras = ring_cameras(n=4)
    scene = build_scene(
        cameras,
        [walking((-1.2, -0.9), (1.2, -0.9), N_FRAMES), standing((1.0, 1.2), N_FRAMES)],
        scales=[1.0, 0.8],
    )
    points, boxes, truth = to_tracked(scene)

    primary = boxes.primary_tracks().df[["cam_id", "sync_index", "track_id", "object_id"]]
    assert {truth[(cam_id, track_id)] for cam_id, track_id in zip(primary["cam_id"], primary["track_id"])} == {SUBJECT}
    assert set(primary["cam_id"]) == set(cameras.cameras)

    # Stands in for running estimate_keypoints on the primary boxes.
    primary_points = ImagePoints(
        points.df.drop(columns=["object_id"]).merge(primary, on=["cam_id", "sync_index", "track_id"])
    )

    run = calibrate_extrinsics(primary_points, scene.intrinsics_only_cameras(), None, refine_intrinsics=False)

    assert set(run.capture_volume.camera_array.posed_cameras) == set(cameras.cameras)
