"""Multicam extraction validated against an independent frame oracle.

Drives the real extract_image_points_multicam (closure, ThreadPoolExecutor, row
flattening) and confirms a sampled (cam, sync_index) observation's detections
match re-detection on the SAME frame decoded independently by ffmpeg. This guards
that the forward iter_frames routing selects and labels the right frames, not a
hand-copied mirror of the production loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from caliscope import __root__
from caliscope.api import Charuco, CharucoTracker, extract_image_points_multicam
from caliscope.helper import copy_contents_to_clean_dest
from tests.oracle_ffmpeg import dump_frame, requires_ffmpeg

CHARUCO_SESSION = Path(__root__, "tests", "sessions", "charuco_calibration")


@requires_ffmpeg
def test_multicam_extraction_matches_independent_oracle(tmp_path: Path):
    copy_contents_to_clean_dest(CHARUCO_SESSION, tmp_path)
    extrinsic = tmp_path / "calibration" / "extrinsic"
    videos = {c: extrinsic / f"cam_{c}.mp4" for c in (0, 1, 2, 3)}
    charuco = Charuco.from_toml(tmp_path / "charuco.toml")
    tracker = CharucoTracker(charuco)

    df = extract_image_points_multicam(
        videos, tracker, frame_step=10, timestamps=extrinsic / "timestamps.csv", progress=None
    ).df
    assert not df.empty

    # Sample one (cam, sync_index) group and re-detect on the independently
    # decoded frame at its frame_index.
    sample = df.iloc[0]
    cam_id, sync_index, frame_index = int(sample["cam_id"]), int(sample["sync_index"]), int(sample["frame_index"])
    group = df[(df["cam_id"] == cam_id) & (df["sync_index"] == sync_index)].sort_values("keypoint_id")

    truth = dump_frame(videos[cam_id], frame_index)
    packet = tracker.get_points(truth, cam_id=cam_id, rotation_count=0)
    order = np.argsort(packet.keypoint_id)

    # Same frame selected -> identical detected corner set.
    assert group["keypoint_id"].tolist() == packet.keypoint_id[order].tolist()
    # img_loc within a pixel: the only difference vs the oracle is YUV->BGR
    # conversion rounding between PyAV and ffmpeg, not a different frame.
    np.testing.assert_allclose(group["img_loc_x"].to_numpy(), packet.img_loc[order, 0], atol=1.0)
    np.testing.assert_allclose(group["img_loc_y"].to_numpy(), packet.img_loc[order, 1], atol=1.0)
