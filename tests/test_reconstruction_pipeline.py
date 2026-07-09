"""End-to-end test of the migrated reconstruction seam.

Mirrors ReconstructionPresenter's worker chain on real calibrated data:
SynchronizedTimestamps.load -> process_synchronized_recording (+ OverlayVideoWriter)
-> reconstruct_xyz. Guards the integration the presenter has no direct test for.
"""

from pathlib import Path

from caliscope import __root__
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.charuco import Charuco
from caliscope.core.process_synchronized_recording import process_synchronized_recording
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.reconstruction.reconstruct_xyz import reconstruct_xyz
from caliscope.recording.overlay_video_writer import OverlayVideoWriter
from caliscope.recording.synchronized_timestamps import SynchronizedTimestamps
from caliscope.trackers.charuco_tracker import CharucoTracker

SESSION = Path(__root__, "tests", "sessions", "post_optimization")


def test_reconstruction_pipeline_produces_xyz_trc_and_overlay(tmp_path: Path):
    copy_contents_to_clean_dest(SESSION, tmp_path)

    camera_array = CameraArray.from_toml(tmp_path / "camera_array.toml")
    tracker = CharucoTracker(Charuco.from_toml(tmp_path / "charuco.toml"))
    recording_dir = tmp_path / "calibration" / "extrinsic"
    cam_ids = sorted(camera_array.cameras.keys())

    synced = SynchronizedTimestamps.load(recording_dir, cam_ids)
    assert synced.mean_fps > 0  # C2: usable rate, never 0/inf

    tracker_dir = recording_dir / tracker.name
    tracker_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 with the overlay sink wired exactly as the presenter wires it.
    recorder = OverlayVideoWriter(tracker_dir, tracker, synced.mean_fps, suffix=tracker.name)
    try:
        image_points = process_synchronized_recording(
            recording_dir=recording_dir,
            cameras=camera_array.cameras,
            tracker=tracker,
            synced_timestamps=synced,
            on_frame_data=recorder.on_frame_data,
            subsample=10,  # keep it quick
        )
    finally:
        recorder.close()

    assert not image_points.df.empty

    # Stage 2.
    reconstruct_xyz(image_points, camera_array, tracker, tracker_dir)

    assert (tracker_dir / f"xyz_{tracker.name}.csv").exists()
    assert (tracker_dir / f"xyz_{tracker.name}_labelled.csv").exists()
    assert (tracker_dir / f"xyz_{tracker.name}.trc").exists()
    for cam_id in cam_ids:
        assert (tracker_dir / f"cam_{cam_id}_{tracker.name}.mp4").exists()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_reconstruction_pipeline_produces_xyz_trc_and_overlay(Path(d))
    print("end-to-end reconstruction pipeline OK")
