"""Tests for recording dimension compatibility assessment."""

from pathlib import Path

import pytest

from caliscope.recording.recording_validation import check_recording_dimensions


EXTRINSIC_DIR = Path(__file__).parent / "sessions" / "4_cam_recording" / "calibration" / "extrinsic"


def test_real_video_dimensions_match_and_preserve_width_height_order() -> None:
    assessment = check_recording_dimensions(EXTRINSIC_DIR, ((0, (1280, 720)),))

    assert assessment.is_compatible
    assert assessment.outcomes[0].actual_size == (1280, 720)
    assert not check_recording_dimensions(EXTRINSIC_DIR, ((0, (720, 1280)),)).is_compatible


def test_checks_sorted_noncontiguous_cameras_and_collects_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    actual_sizes = {2: (1280, 720), 7: (1920, 1080), 11: (640, 480)}

    def fake_read(path: Path) -> tuple[int, int]:
        cam_id = int(path.stem.removeprefix("cam_"))
        seen.append(path.name)
        return actual_sizes[cam_id]

    monkeypatch.setattr("caliscope.recording.recording_validation.read_video_dimensions", fake_read)

    assessment = check_recording_dimensions(
        tmp_path,
        ((11, (640, 480)), (2, (1920, 1080)), (7, (1920, 1080))),
    )

    assert [outcome.cam_id for outcome in assessment.outcomes] == [2, 7, 11]
    assert seen == ["cam_2.mp4", "cam_7.mp4", "cam_11.mp4"]
    assert assessment.outcomes[0].actual_size == (1280, 720)
    assert assessment.outcomes[0].error is None
    assert assessment.outcomes[1].actual_size == (1920, 1080)
    assert assessment.outcomes[2].actual_size == (640, 480)
    assert not assessment.is_compatible


def test_reports_media_and_calibration_errors_for_every_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read(path: Path) -> tuple[int, int]:
        raise ValueError("no video stream")

    monkeypatch.setattr("caliscope.recording.recording_validation.read_video_dimensions", fake_read)

    assessment = check_recording_dimensions(tmp_path, ((3, None), (9, (640, 480))))

    invalid_calibration, unreadable_video = assessment.outcomes
    assert invalid_calibration.cam_id == 3
    assert invalid_calibration.actual_size is None
    assert invalid_calibration.error == "Calibration dimensions are missing or invalid for camera 3."
    assert unreadable_video.cam_id == 9
    assert unreadable_video.actual_size is None
    assert "Could not read video dimensions" in (unreadable_video.error or "")
    assert not assessment.is_compatible


def test_empty_expected_set_is_not_compatible(tmp_path: Path) -> None:
    assert not check_recording_dimensions(tmp_path, ()).is_compatible
