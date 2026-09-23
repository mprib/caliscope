"""Tests for ReconstructionPresenter.

Tasks are driven through real TaskHandles from FakeTaskManager: the dimension
check ("recording-dimension-check") and the reconstruction ("reconstruction").
"""

import copy
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from caliscope import __root__
from caliscope.cameras.camera_array import CameraArray
from caliscope.gui.presenters.reconstruction_presenter import (
    ReconstructionPresenter,
    ReconstructionState,
)
from caliscope.helper import copy_contents_to_clean_dest
from caliscope.recording.recording_validation import (
    CameraDimensionOutcome,
    RecordingDimensionAssessment,
)
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.trackers import tracker_registry
from caliscope.workspace_guide import WorkspaceGuide

TEST_SESSION = Path(__root__) / "tests" / "sessions" / "4_cam_recording"
DIMENSION_CHECK = "recording-dimension-check"
RESTART_NOTICE = "Recording check restarted. Select Process when it finishes."


def _create_recording_session(workspace: Path, name: str) -> Path:
    session = workspace / "recordings" / name
    session.mkdir()
    for cam_id in (0, 1, 2, 3):
        (session / f"cam_{cam_id}.mp4").touch()
    return session


def _compatible_dimensions(camera_array: CameraArray) -> RecordingDimensionAssessment:
    return RecordingDimensionAssessment(
        tuple(
            CameraDimensionOutcome(cam_id, camera.size, camera.size, None)
            for cam_id, camera in sorted(camera_array.cameras.items())
        )
    )


def _mismatched_dimensions(camera_array: CameraArray) -> RecordingDimensionAssessment:
    """One exact-size mismatch, with outcomes for every camera."""
    first = min(camera_array.cameras)
    return RecordingDimensionAssessment(
        tuple(
            CameraDimensionOutcome(
                cam_id, camera.size, (camera.size[0] - 1, camera.size[1]) if cam_id == first else camera.size, None
            )
            for cam_id, camera in sorted(camera_array.cameras.items())
        )
    )


@pytest.fixture
def workspace(tmp_path):
    copy_contents_to_clean_dest(TEST_SESSION, tmp_path)
    return tmp_path


@pytest.fixture
def recording(workspace) -> Path:
    return workspace / "recordings" / "recording_1"


@pytest.fixture
def camera_array(workspace):
    return CameraArray.from_toml(workspace / "camera_array.toml")


@pytest.fixture
def workspace_guide(workspace):
    return WorkspaceGuide(workspace)


@pytest.fixture
def presenter(workspace, workspace_guide, camera_array, fake_task_manager):
    return ReconstructionPresenter(
        workspace_dir=workspace,
        workspace_guide=workspace_guide,
        camera_array=camera_array,
        task_manager=fake_task_manager,
    )


@pytest.fixture
def finish_check(fake_task_manager, camera_array, qapp):
    """Complete the newest dimension check, compatible unless told otherwise."""

    def finish(assessment: RecordingDimensionAssessment | None = None) -> TaskHandle:
        handle = fake_task_manager.handles(DIMENSION_CHECK)[-1]
        fake_task_manager.complete(handle, assessment or _compatible_dimensions(camera_array))
        qapp.processEvents()
        return handle

    return finish


@pytest.fixture
def submit_reconstruction(presenter, fake_task_manager, finish_check, registered_tracker):
    """Select recording_1, pass both dimension checks, and return the PENDING reconstruction handle."""

    def submit() -> TaskHandle:
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)
        finish_check()
        presenter.start_reconstruction()
        finish_check()
        (handle,) = fake_task_manager.handles("reconstruction")
        return handle

    return submit


class TestStateComputation:
    def test_initial_state_is_idle(self, presenter):
        assert presenter.state == ReconstructionState.IDLE

    @pytest.mark.parametrize(
        ("finish", "expected"),
        [
            pytest.param(None, ReconstructionState.RECONSTRUCTING, id="pending"),
            pytest.param("start", ReconstructionState.RECONSTRUCTING, id="running"),
            pytest.param("fail", ReconstructionState.ERROR, id="failed"),
            pytest.param("cancel", ReconstructionState.IDLE, id="cancelled"),
        ],
    )
    def test_task_state_maps_to_presenter_state(
        self, presenter, submit_reconstruction, fake_task_manager, qapp, finish, expected
    ):
        handle = submit_reconstruction()
        if finish is not None:
            getattr(fake_task_manager, finish)(handle)
            qapp.processEvents()

        assert presenter.state == expected

    def test_state_error_when_start_is_rejected(self, presenter):
        presenter.start_reconstruction()

        assert presenter.last_error is not None
        assert presenter.state == ReconstructionState.ERROR

    def test_state_complete_when_xyz_exists(self, presenter, recording, registered_tracker):
        """Existing output remains complete even if its source recording degrades."""
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)
        output_dir = recording / registered_tracker
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"xyz_{registered_tracker}.csv").touch()
        (recording / "cam_2.mp4").unlink()

        assert presenter.state == ReconstructionState.COMPLETE
        assert presenter.selected_recording_is_ready is False

    def test_task_state_takes_precedence_over_file(
        self, presenter, submit_reconstruction, fake_task_manager, recording, registered_tracker
    ):
        handle = submit_reconstruction()
        fake_task_manager.start(handle)
        (recording / registered_tracker / f"xyz_{registered_tracker}.csv").touch()

        assert presenter.state == ReconstructionState.RECONSTRUCTING


class TestTaskSignals:
    """Handle signals reach the presenter through its queued connections."""

    def test_progress_and_completion(
        self, presenter, submit_reconstruction, fake_task_manager, qapp, recording, registered_tracker
    ):
        progress = []
        completed = []
        presenter.progress_updated.connect(lambda percent, message: progress.append((percent, message)))
        presenter.reconstruction_complete.connect(completed.append)
        handle = submit_reconstruction()

        fake_task_manager.start(handle)
        handle.report_progress(40, "Stage 1: 50% - Detecting 2D landmarks")
        qapp.processEvents()

        assert progress == [(40, "Stage 1: 50% - Detecting 2D landmarks")]
        assert presenter.state == ReconstructionState.RECONSTRUCTING

        tracker_dir = recording / registered_tracker
        (tracker_dir / f"xyz_{registered_tracker}.csv").touch()
        fake_task_manager.complete(handle, tracker_dir)
        qapp.processEvents()

        assert completed == [tracker_dir / f"xyz_{registered_tracker}.csv"]
        assert presenter.state == ReconstructionState.COMPLETE


class TestAvailableOptions:
    def test_available_recordings_includes_invalid_session_directories(self, presenter, workspace):
        """The list mirrors immediate session directories instead of filtering them."""
        (workspace / "recordings" / "empty_session").mkdir()

        recordings = presenter.available_recordings

        assert "recording_1" in recordings
        assert "empty_session" in recordings


class TestSelection:
    def test_select_recording(self, presenter):
        presenter.select_recording("recording_1")
        assert presenter.selected_recording == "recording_1"

    def test_select_tracker(self, presenter, registered_tracker):
        presenter.select_tracker(registered_tracker)
        assert presenter.selected_tracker == registered_tracker

    def test_selection_clears_error(self, presenter, registered_tracker):
        presenter.start_reconstruction()
        assert presenter.last_error is not None
        presenter.select_recording("recording_1")
        assert presenter.last_error is None

        presenter.start_reconstruction()
        assert presenter.last_error is not None
        presenter.select_tracker(registered_tracker)
        assert presenter.last_error is None

    def test_select_structurally_invalid_recording_shows_its_assessment(self, presenter, workspace):
        """Invalid session rows remain selectable so their feedback is visible."""
        invalid = workspace / "recordings" / "partial"
        invalid.mkdir()
        (invalid / "cam_0.mp4").touch()

        presenter.select_recording("partial")

        assert presenter.selected_recording == "partial"
        assert presenter.selected_recording_is_ready is False
        assert presenter.selected_recording_assessment is not None
        assert presenter.selected_recording_assessment.missing_camera_ids == (1, 2, 3)

    def test_nonexistent_recording_is_ignored(self, presenter):
        presenter.select_recording("nonexistent")
        assert presenter.selected_recording is None

    def test_refresh_preserves_selection_and_recovers_after_video_restore(
        self, presenter, recording, registered_tracker
    ):
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)

        (recording / "cam_2.mp4").unlink()
        presenter.refresh_from_workspace()

        assert presenter.selected_recording == "recording_1"
        assert presenter.can_process is False
        assert presenter.selected_recording_assessment is not None
        assert [issue.relative_path for issue in presenter.selected_recording_assessment.issues] == [
            "recordings/recording_1/cam_2.mp4"
        ]

        (recording / "cam_2.mp4").touch()
        presenter.refresh_from_workspace()

        assert presenter.selected_recording == "recording_1"
        assert presenter.is_checking_dimensions is True
        assert presenter.can_process is False

    def test_disappearing_session_suppresses_stale_dimension_errors(self, presenter, recording, finish_check):
        """The structural missing-directory issue is enough when a scan finishes late."""
        presenter.select_recording("recording_1")
        finish_check(RecordingDimensionAssessment((CameraDimensionOutcome(0, (1280, 720), None, "video disappeared"),)))
        shutil.rmtree(recording)

        assert [issue.code for issue in presenter.selected_recording_issues] == ["missing_recording_directory"]
        assert presenter.selected_recording_dimension_issues == ()

    @pytest.mark.parametrize(
        ("outcome", "expected_code"),
        [
            (CameraDimensionOutcome(0, (1280, 720), (640, 480), None), "recording_dimension_mismatch"),
            (CameraDimensionOutcome(0, (1280, 720), None, "unopenable"), "unreadable_recording_dimensions"),
        ],
    )
    def test_dimension_outcomes_block_processing_with_actionable_feedback(
        self, presenter, registered_tracker, finish_check, outcome, expected_code
    ):
        """Mismatch and read failure both remain visible and block Process."""
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)
        finish_check(RecordingDimensionAssessment((outcome,)))

        assert presenter.can_process is False
        issues = presenter.selected_recording_dimension_issues
        assert [issue.code for issue in issues] == [expected_code]
        assert issues[0].relative_path == "recordings/recording_1/cam_0.mp4"

    @pytest.mark.parametrize("running", [False, True], ids=["pending", "running"])
    def test_refresh_reports_removed_active_session_without_cancelling(
        self, presenter, submit_reconstruction, fake_task_manager, recording, running
    ):
        handle = submit_reconstruction()
        if running:
            fake_task_manager.start(handle)
        shutil.rmtree(recording)

        presenter.refresh_from_workspace()

        assert presenter.selected_recording == "recording_1"
        assert [issue.code for issue in presenter.selected_recording_issues] == ["missing_recording_directory"]
        assert not fake_task_manager.cancel_requested(handle)


class TestSignalEmissions:
    def test_state_changed_emitted_on_selection(self, presenter):
        states = []
        presenter.state_changed.connect(states.append)

        presenter.select_recording("recording_1")

        assert states[-1] == ReconstructionState.IDLE

    def test_state_changed_emitted_on_tracker_selection(self, presenter, registered_tracker):
        states = []
        presenter.state_changed.connect(states.append)

        presenter.select_tracker(registered_tracker)

        assert states

    def test_dimension_completion_updates_state_without_rebuilding_recording_list(self, presenter, finish_check):
        """Dimension results update feedback controls, not the recording list or preview."""
        recordings_changed = []
        states = []
        presenter.recordings_changed.connect(lambda: recordings_changed.append(True))
        presenter.state_changed.connect(states.append)
        presenter.select_recording("recording_1")
        states.clear()

        finish_check()

        assert recordings_changed == []
        assert states


class TestScopedWorkspaceRefresh:
    def test_selected_assessment_reads_only_selected_directory(self, presenter, workspace_guide, monkeypatch):
        presenter.select_recording("recording_1")
        all_recordings = MagicMock(side_effect=AssertionError("selected feedback must not scan every session"))
        monkeypatch.setattr(workspace_guide, "assess_recordings", all_recordings)

        assessment = presenter.selected_recording_assessment

        assert assessment is not None
        all_recordings.assert_not_called()

    def test_other_recording_changes_do_not_request_metadata(
        self, presenter, workspace, fake_task_manager, finish_check
    ):
        other = _create_recording_session(workspace, "other")
        presenter.select_recording("recording_1")
        finish_check()
        submitted = len(fake_task_manager.submitted)

        presenter.refresh_recording_structure(other)
        presenter.recheck_selected_video(other / "cam_0.mp4")

        assert presenter.selected_recording_is_ready
        assert len(fake_task_manager.submitted) == submitted

    def test_same_size_calibration_change_cancels_process_preflight(
        self, presenter, registered_tracker, fake_task_manager, finish_check, camera_array, qapp
    ):
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)
        finish_check()
        presenter.start_reconstruction()
        process_check = fake_task_manager.handles(DIMENSION_CHECK)[-1]
        changed_calibration = copy.deepcopy(camera_array)
        assert changed_calibration.cameras[0].matrix is not None
        changed_calibration.cameras[0].matrix += np.eye(3)

        presenter.refresh_camera_array(changed_calibration)
        fake_task_manager.complete(process_check, _compatible_dimensions(camera_array))
        qapp.processEvents()

        assert fake_task_manager.cancel_requested(process_check)
        assert presenter.dimension_check_notice == RESTART_NOTICE
        assert fake_task_manager.handles("reconstruction") == []

    def test_tracker_selection_cancels_process_continuation(
        self, presenter, registered_tracker, fake_task_manager, finish_check, camera_array, qapp
    ):
        presenter.select_recording("recording_1")
        finish_check()
        presenter.select_tracker(registered_tracker)
        assert presenter.selected_recording_is_ready
        presenter.start_reconstruction()
        process_check = fake_task_manager.handles(DIMENSION_CHECK)[-1]

        presenter.select_tracker(registered_tracker)
        fake_task_manager.complete(process_check, _compatible_dimensions(camera_array))
        qapp.processEvents()

        assert fake_task_manager.cancel_requested(process_check)
        assert fake_task_manager.handles("reconstruction") == []


class TestStartReconstruction:
    def test_cannot_start_without_selection(self, presenter, fake_task_manager):
        presenter.start_reconstruction()

        assert presenter.last_error is not None
        assert fake_task_manager.submitted == []

    def test_start_submits_reconstruction_after_fresh_preflight(
        self, presenter, fake_task_manager, finish_check, registered_tracker
    ):
        """A Process click submits a fresh check before the reconstruction task."""
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)
        finish_check()
        presenter.start_reconstruction()
        started_at_signal = []
        presenter.reconstruction_starting.connect(lambda: started_at_signal.extend(fake_task_manager.started))

        finish_check()

        (handle,) = fake_task_manager.handles("reconstruction")
        assert handle.task_id not in started_at_signal
        assert fake_task_manager.started[-1] == handle.task_id

    def test_incompatible_process_preflight_has_no_output_side_effects(
        self, presenter, recording, fake_task_manager, finish_check, camera_array, registered_tracker, monkeypatch
    ):
        """Fresh Process dimensions failure stops before timestamps, tracker, and output."""
        load_timestamps = MagicMock()
        create_tracker = MagicMock()
        monkeypatch.setattr(
            "caliscope.gui.presenters.reconstruction_presenter.SynchronizedTimestamps.load",
            load_timestamps,
        )
        monkeypatch.setattr(tracker_registry, "create", create_tracker)
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)
        finish_check()
        presenter.start_reconstruction()
        starts = []
        presenter.reconstruction_starting.connect(lambda: starts.append(True))

        finish_check(_mismatched_dimensions(camera_array))

        load_timestamps.assert_not_called()
        create_tracker.assert_not_called()
        assert not (recording / registered_tracker).exists()
        assert fake_task_manager.handles("reconstruction") == []
        assert starts == []

    def test_missing_camera_video_blocks_submission_even_with_timestamps(
        self, presenter, recording, fake_task_manager, registered_tracker
    ):
        assert (recording / "timestamps.csv").exists()
        (recording / "cam_3.mp4").unlink()
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)

        presenter.start_reconstruction()

        assert fake_task_manager.submitted == []
        assert presenter.selected_recording_is_ready is False

    def test_timestamp_failure_creates_no_tracker_output_or_task(
        self, presenter, recording, fake_task_manager, finish_check, registered_tracker, monkeypatch
    ):
        load_timestamps = MagicMock(side_effect=ValueError("Timestamp data is incompatible"))
        create_tracker = MagicMock()
        monkeypatch.setattr(
            "caliscope.gui.presenters.reconstruction_presenter.SynchronizedTimestamps.load",
            load_timestamps,
        )
        monkeypatch.setattr(tracker_registry, "create", create_tracker)
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)
        finish_check()

        presenter.start_reconstruction()
        finish_check()

        load_timestamps.assert_called_once()
        create_tracker.assert_not_called()
        assert not (recording / registered_tracker).exists()
        assert fake_task_manager.handles("reconstruction") == []

    def test_repeated_start_does_not_submit_while_first_task_is_pending(
        self, presenter, submit_reconstruction, fake_task_manager
    ):
        submit_reconstruction()

        presenter.start_reconstruction()

        assert len(fake_task_manager.handles("reconstruction")) == 1

    def test_invalidating_process_preflight_drops_automatic_continuation(
        self, presenter, fake_task_manager, finish_check, camera_array, registered_tracker, qapp
    ):
        """A changed session cannot continue into output creation after Process."""
        presenter.select_recording("recording_1")
        presenter.select_tracker(registered_tracker)
        finish_check()
        presenter.start_reconstruction()
        process_check = fake_task_manager.handles(DIMENSION_CHECK)[-1]

        presenter.refresh_from_workspace()
        fake_task_manager.complete(process_check, _compatible_dimensions(camera_array))
        qapp.processEvents()

        assert fake_task_manager.handles("reconstruction") == []
        assert presenter.dimension_check_notice == RESTART_NOTICE

        finish_check()
        presenter.start_reconstruction()

        assert presenter.dimension_check_notice is None


class TestTerminalSelectionReconciliation:
    def test_completion_emits_original_result_before_selecting_remaining_session(
        self, presenter, submit_reconstruction, fake_task_manager, workspace, recording, registered_tracker, qapp
    ):
        remaining = _create_recording_session(workspace, "remaining")
        handle = submit_reconstruction()
        shutil.rmtree(recording)
        result_dir = workspace / "completed-result"
        observed: list[tuple[Path, str | None]] = []
        presenter.reconstruction_complete.connect(lambda path: observed.append((path, presenter.selected_recording)))

        fake_task_manager.complete(handle, result_dir)
        qapp.processEvents()

        assert observed == [(result_dir / f"xyz_{registered_tracker}.csv", "recording_1")]
        assert presenter.selected_recording == remaining.name

    def test_failure_reconciles_removed_selection_after_failure_signal(
        self, presenter, submit_reconstruction, fake_task_manager, workspace, recording, qapp
    ):
        remaining = _create_recording_session(workspace, "remaining")
        handle = submit_reconstruction()
        shutil.rmtree(recording)
        observed: list[tuple[str, str | None]] = []
        presenter.reconstruction_failed.connect(
            lambda message: observed.append((message, presenter.selected_recording))
        )

        fake_task_manager.fail(handle, "RuntimeError", "worker failed")
        qapp.processEvents()

        assert observed == [("RuntimeError: worker failed", "recording_1")]
        assert presenter.selected_recording == remaining.name
        assert presenter.last_error == "RuntimeError: worker failed"
        assert presenter.state == ReconstructionState.ERROR

    def test_cancellation_reconciles_removed_selection(
        self, presenter, submit_reconstruction, fake_task_manager, workspace, recording, qapp
    ):
        remaining = _create_recording_session(workspace, "remaining")
        handle = submit_reconstruction()
        shutil.rmtree(recording)

        fake_task_manager.cancel(handle)
        qapp.processEvents()

        assert presenter.selected_recording == remaining.name
        assert presenter.state == ReconstructionState.IDLE


class TestCleanup:
    def test_cleanup_cancels_running_task(self, presenter, submit_reconstruction, fake_task_manager):
        handle = submit_reconstruction()
        fake_task_manager.start(handle)

        presenter.cleanup()

        assert fake_task_manager.cancel_requested(handle)

    def test_cleanup_safe_when_no_task(self, presenter):
        presenter.cleanup()

    def test_cancel_accepts_pending_task(self, presenter, submit_reconstruction, fake_task_manager):
        """A task can be cancelled before its worker thread begins."""
        handle = submit_reconstruction()

        presenter.cancel_reconstruction()

        assert fake_task_manager.cancel_requested(handle)
