"""Tests for caliscope.reporting — sparklines, badges, progress, and print functions."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path


from caliscope import __root__
from caliscope.reporting import (
    ProgressCallback,
    RichProgressBar,
    _quality_badge,
    _sparkline,
    print_extrinsic_report,
    print_intrinsic_report,
)

logger = logging.getLogger(__name__)

PRERECORDED_SESSION = Path(__root__, "tests", "sessions", "prerecorded_calibration")
POST_OPTIMIZATION_SESSION = Path(__root__, "tests", "sessions", "post_optimization")


# ---------------------------------------------------------------------------
# _sparkline
# ---------------------------------------------------------------------------


def test_sparkline_empty_list():
    """Empty input produces an empty string."""
    assert _sparkline([]) == ""


def test_sparkline_single_value():
    """A single value produces a single character."""
    result = _sparkline([5.0])
    assert len(result) == 1


def test_sparkline_all_equal():
    """All-equal values produce the same character repeated."""
    result = _sparkline([1.0, 1.0, 1.0])
    assert len(result) == 3
    # All characters must be identical
    assert len(set(result)) == 1


def test_sparkline_increasing():
    """Increasing values should produce non-decreasing block heights."""
    BLOCKS = " ▁▂▃▄▅▆▇█"

    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    result = _sparkline(values)

    assert len(result) == len(values)

    # Each character's block index should be >= the previous one
    indices = [BLOCKS.index(ch) for ch in result]
    for i in range(1, len(indices)):
        assert indices[i] >= indices[i - 1], (
            f"Expected non-decreasing bar heights for increasing input, "
            f"but position {i} ({result[i]!r}) is less than position {i - 1} ({result[i - 1]!r})"
        )


# ---------------------------------------------------------------------------
# _quality_badge
# ---------------------------------------------------------------------------


def test_quality_badge_thresholds():
    """Badge labels and colors must match the documented threshold boundaries."""
    thresholds = [
        (0.5, "EXCELLENT", "green"),
        (1.0, "GOOD", "yellow"),
        (float("inf"), "POOR", "red"),
    ]

    # Strictly below first threshold
    assert _quality_badge(0.0, thresholds) == "[green]EXCELLENT[/green]"
    assert _quality_badge(0.499, thresholds) == "[green]EXCELLENT[/green]"

    # At first threshold boundary (not strictly less than)
    assert _quality_badge(0.5, thresholds) == "[yellow]GOOD[/yellow]"

    # Between first and second
    assert _quality_badge(0.75, thresholds) == "[yellow]GOOD[/yellow]"
    assert _quality_badge(0.999, thresholds) == "[yellow]GOOD[/yellow]"

    # At second threshold boundary
    assert _quality_badge(1.0, thresholds) == "[red]POOR[/red]"

    # Above second threshold
    assert _quality_badge(5.0, thresholds) == "[red]POOR[/red]"


# ---------------------------------------------------------------------------
# RichProgressBar — protocol conformance
# ---------------------------------------------------------------------------


def test_rich_progress_bar_protocol():
    """RichProgressBar must satisfy the ProgressCallback protocol."""
    assert isinstance(RichProgressBar(), ProgressCallback)


# ---------------------------------------------------------------------------
# print_intrinsic_report — smoke test
# ---------------------------------------------------------------------------


def test_print_intrinsic_report_no_crash():
    """print_intrinsic_report must not raise when given a real calibration output."""
    from rich.console import Console

    from caliscope.api import (
        CameraData,
        CharucoTracker,
        Charuco,
        calibrate_intrinsics,
        extract_image_points,
    )

    charuco = Charuco.from_toml(PRERECORDED_SESSION / "charuco.toml")
    tracker = CharucoTracker(charuco)

    video_path = PRERECORDED_SESSION / "calibration" / "intrinsic" / "cam_0.mp4"
    image_points = extract_image_points({0: video_path}, tracker)

    camera = CameraData(cam_id=0, size=(1280, 720))
    output = calibrate_intrinsics(image_points, camera)

    sink = StringIO()
    console = Console(file=sink, highlight=False)

    # Should not raise
    print_intrinsic_report(output, console=console)

    printed = sink.getvalue()
    assert "Intrinsic Calibration" in printed
    assert "RMSE" in printed


# ---------------------------------------------------------------------------
# print_extrinsic_report — smoke test
# ---------------------------------------------------------------------------


def test_print_extrinsic_report_no_crash():
    """print_extrinsic_report must not raise when given a real CaptureVolume."""
    from rich.console import Console

    from caliscope.cameras.camera_array import CameraArray
    from caliscope.core.capture_volume import CaptureVolume
    from caliscope.core.point_data import ImagePoints, WorldPoints

    image_points_path = POST_OPTIMIZATION_SESSION / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
    camera_array_path = POST_OPTIMIZATION_SESSION / "camera_array.toml"
    world_points_path = POST_OPTIMIZATION_SESSION / "calibration" / "extrinsic" / "CHARUCO" / "xyz_CHARUCO.csv"

    camera_array = CameraArray.from_toml(camera_array_path)
    image_points = ImagePoints.from_csv(image_points_path)
    world_points = WorldPoints.from_csv(world_points_path)

    capture_volume = CaptureVolume(
        camera_array=camera_array,
        image_points=image_points,
        world_points=world_points,
    )

    sink = StringIO()
    console = Console(file=sink, highlight=False)

    # Should not raise
    print_extrinsic_report(capture_volume, console=console)

    printed = sink.getvalue()
    assert "Extrinsic Calibration" in printed


# ---------------------------------------------------------------------------
# Debug harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    from caliscope.logger import setup_logging

    setup_logging()

    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    logger.info("test_sparkline_empty_list")
    test_sparkline_empty_list()

    logger.info("test_sparkline_single_value")
    test_sparkline_single_value()

    logger.info("test_sparkline_all_equal")
    test_sparkline_all_equal()

    logger.info("test_sparkline_increasing")
    test_sparkline_increasing()

    logger.info("test_quality_badge_thresholds")
    test_quality_badge_thresholds()

    logger.info("test_rich_progress_bar_protocol")
    test_rich_progress_bar_protocol()

    logger.info("test_print_intrinsic_report_no_crash")
    test_print_intrinsic_report_no_crash()

    logger.info("test_print_extrinsic_report_no_crash")
    test_print_extrinsic_report_no_crash()

    logger.info("All reporting tests passed.")
