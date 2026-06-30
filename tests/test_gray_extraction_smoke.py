from pathlib import Path

import pytest

from caliscope.api import extract_image_points


CHARUCO_SESSION = Path("tests/sessions/charuco_calibration")


@pytest.fixture
def charuco_video_path() -> Path:
    """Path to a test video with visible charuco board."""
    p = CHARUCO_SESSION / "calibration" / "extrinsic"
    videos = sorted(p.glob("cam_*.mp4"))
    if not videos:
        pytest.skip("No charuco calibration test video available")
    return videos[0]


def test_gray_extraction_produces_points(charuco_video_path: Path):
    """extract_image_points works end-to-end with a GRAY tracker.

    Loads the charuco config from the test session's TOML so the board
    params match the video content.
    """
    from caliscope.core.charuco import Charuco
    from caliscope.trackers.charuco_tracker import CharucoTracker

    charuco_toml = CHARUCO_SESSION / "charuco.toml"
    if not charuco_toml.exists():
        pytest.skip("No charuco.toml in test session")
    charuco = Charuco.from_toml(charuco_toml)
    tracker = CharucoTracker(charuco)
    cam_id = 0

    points = extract_image_points(charuco_video_path, cam_id, tracker, frame_step=5, progress=None)
    assert len(points.df) > 0
    assert "img_loc_x" in points.df.columns
    assert "img_loc_y" in points.df.columns
