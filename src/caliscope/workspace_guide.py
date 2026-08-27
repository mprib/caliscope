import logging
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.workflow_status import WorkspaceIssue

logger = logging.getLogger(__name__)

_CAMERA_VIDEO_PATTERN = re.compile(r"^cam_(0|[1-9][0-9]*)\.mp4$")
_CAMERA_DIRECTORY_PATTERN = re.compile(r"^cam_(0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class CameraVideoAssessment:
    """Snapshot of direct-child camera videos in one directory."""

    camera_ids: tuple[int, ...]
    missing_camera_ids: tuple[int, ...]
    issues: tuple[WorkspaceIssue, ...]

    @property
    def is_ready(self) -> bool:
        """Whether the directory contains a complete, unambiguous camera set."""
        return bool(self.camera_ids) and not self.missing_camera_ids and not self.issues


class WorkspaceGuide:
    """
    Utility class for inspecting workspace directory structure and reporting
    on calibration workflow status. This class maintains NO domain state -
    it receives current state from the Controller and reports on filesystem state.
    """

    def __init__(self, workspace_dir: Path) -> None:
        """
        Args:
            workspace_dir: Root workspace directory path
        """
        self.workspace_dir = workspace_dir
        self.calibration_dir = Path(workspace_dir, "calibration")
        self.intrinsic_dir = Path(workspace_dir, "calibration", "intrinsic")
        self.extrinsic_dir = Path(workspace_dir, "calibration", "extrinsic")
        self.recording_dir = Path(workspace_dir, "recordings")

    def get_cam_ids_in_dir(self, directory: Path) -> list[int]:
        """
        Return list of camera IDs from video files in directory.

        Args:
            directory: Path to scan for cam_N.mp4 files

        Returns:
            Sorted list of integer camera IDs found
        """
        return list(self.assess_camera_videos(directory, ()).camera_ids)

    def assess_intrinsic_videos(self, expected_cam_ids: Collection[int]) -> CameraVideoAssessment:
        """Assess intrinsic videos against the workspace camera set."""
        return self.assess_camera_videos(self.intrinsic_dir, expected_cam_ids)

    def assess_extrinsic_videos(self, expected_cam_ids: Collection[int]) -> CameraVideoAssessment:
        """Assess extrinsic videos against the workspace camera set.

        Per-camera missing issues cover any expected camera. The generic
        "no videos" issue only applies when there is nothing more specific to say.
        """
        assessment = self.assess_camera_videos(self.extrinsic_dir, expected_cam_ids)
        if assessment.camera_ids or assessment.missing_camera_ids:
            return assessment

        issue = WorkspaceIssue(
            code="missing_camera_video",
            message="No extrinsic camera videos found. Add cam_N.mp4 files directly to calibration/extrinsic/.",
            relative_path="calibration/extrinsic",
        )
        return CameraVideoAssessment(
            camera_ids=assessment.camera_ids,
            missing_camera_ids=assessment.missing_camera_ids,
            issues=(issue, *assessment.issues),
        )

    def assess_camera_videos(
        self,
        directory: Path,
        expected_cam_ids: Collection[int],
    ) -> CameraVideoAssessment:
        """Assess direct-child cam_N.mp4 files in any directory against a camera set.

        Reports missing expected videos, videos outside the expected set, and
        MP4 files that do not follow the cam_N.mp4 naming.
        """
        camera_ids: list[int] = []
        file_issues: list[WorkspaceIssue] = []

        if directory.exists():
            for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
                if not path.is_file():
                    continue

                match = _CAMERA_VIDEO_PATTERN.fullmatch(path.name)
                if match is not None:
                    camera_ids.append(int(match.group(1)))
                    continue

                if path.suffix.casefold() != ".mp4":
                    continue

                relative_path = self._relative_path(path)
                if path.name.casefold().startswith("cam"):
                    file_issues.append(
                        WorkspaceIssue(
                            code="malformed_camera_filename",
                            message=f"{relative_path} looks like a camera video but must be named cam_N.mp4.",
                            relative_path=relative_path,
                        )
                    )
                else:
                    file_issues.append(
                        WorkspaceIssue(
                            code="unexpected_mp4",
                            message=f"{relative_path} is an unexpected MP4. Only cam_N.mp4 files belong here.",
                            relative_path=relative_path,
                        )
                    )

        actual_ids = set(camera_ids)
        expected_ids = set(expected_cam_ids)
        missing_ids = tuple(sorted(expected_ids - actual_ids))
        missing_issues = tuple(
            WorkspaceIssue(
                code="missing_camera_video",
                message=f"Missing {self._relative_path(directory / f'cam_{cam_id}.mp4')}.",
                relative_path=self._relative_path(directory / f"cam_{cam_id}.mp4"),
            )
            for cam_id in missing_ids
        )

        unexpected_issues: list[WorkspaceIssue] = []
        if expected_ids:
            for cam_id in sorted(actual_ids - expected_ids):
                relative_path = self._relative_path(directory / f"cam_{cam_id}.mp4")
                unexpected_issues.append(
                    WorkspaceIssue(
                        code="unexpected_mp4",
                        message=f"{relative_path} does not match the workspace camera set.",
                        relative_path=relative_path,
                    )
                )

        return CameraVideoAssessment(
            camera_ids=tuple(sorted(actual_ids)),
            missing_camera_ids=missing_ids,
            issues=(*missing_issues, *unexpected_issues, *file_issues),
        )

    def recording_layout_issues(self) -> tuple[WorkspaceIssue, ...]:
        """Report recognizable recording layouts that Caliscope cannot use."""
        if not self.recording_dir.exists():
            return ()

        children = tuple(self.recording_dir.iterdir())
        root_mp4s = sorted(path.name for path in children if path.is_file() and path.suffix.casefold() == ".mp4")
        root_timestamps = sorted(
            path.name for path in children if path.is_file() and path.name.casefold() == "timestamps.csv"
        )
        camera_named_directories = [
            path for path in children if path.is_dir() and _CAMERA_DIRECTORY_PATTERN.fullmatch(path.name)
        ]
        noncanonical_camera_directories: set[str] = set()
        canonical_partition_directories: set[str] = set()
        for path in camera_named_directories:
            mp4_names = {
                child.name for child in path.iterdir() if child.is_file() and child.suffix.casefold() == ".mp4"
            }
            if any(_CAMERA_VIDEO_PATTERN.fullmatch(name) is None for name in mp4_names):
                noncanonical_camera_directories.add(path.name)
            if mp4_names == {f"{path.name}.mp4"}:
                canonical_partition_directories.add(path.name)

        camera_directories = noncanonical_camera_directories
        if len(canonical_partition_directories) >= 2:
            camera_directories |= canonical_partition_directories

        issues: list[WorkspaceIssue] = []
        if root_mp4s:
            file_list = ", ".join(f"recordings/{name}" for name in root_mp4s)
            issues.append(
                WorkspaceIssue(
                    code="recordings_need_session_folder",
                    message=(
                        f"{file_list} must be inside a named recording session folder. "
                        "Each session folder should contain its cam_N.mp4 files."
                    ),
                    relative_path="recordings",
                )
            )
        if root_timestamps:
            file_list = ", ".join(f"recordings/{name}" for name in root_timestamps)
            issues.append(
                WorkspaceIssue(
                    code="recording_timestamps_need_session_folder",
                    message=f"{file_list} must be inside the recording session folder it describes.",
                    relative_path="recordings",
                )
            )
        if camera_directories:
            directory_list = ", ".join(f"recordings/{name}" for name in sorted(camera_directories))
            issues.append(
                WorkspaceIssue(
                    code="recording_split_by_camera",
                    message=(
                        f"{directory_list} appears to split a recording by camera. "
                        "Create one named session folder containing every cam_N.mp4 file instead."
                    ),
                    relative_path="recordings",
                )
            )
        return tuple(issues)

    def assess_recordings(
        self,
        expected_cam_ids: Collection[int],
    ) -> dict[str, CameraVideoAssessment]:
        """Assess every immediate recording session directory.

        Session names are unconstrained. Empty and structurally invalid
        directories remain in the result so the GUI can show and explain them.
        A timestamps.csv file is optional and is ignored by camera assessment.
        """
        if not self.recording_dir.exists():
            return {}

        return {
            path.name: self.assess_camera_videos(path, expected_cam_ids)
            for path in sorted(self.recording_dir.iterdir(), key=lambda item: item.name.casefold())
            if path.is_dir()
        }

    def ready_recording_dirs(self, expected_cam_ids: Collection[int]) -> list[str]:
        """Return recording session names with a complete camera video set."""
        if not expected_cam_ids:
            return []
        assessments = self.assess_recordings(expected_cam_ids)
        return [name for name, assessment in assessments.items() if assessment.is_ready]

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.workspace_dir).as_posix()

    def get_cam_ids(self) -> list[int]:
        """Return the authoritative list of camera IDs from extrinsic directory.

        The extrinsic directory is the source of truth for the camera set because:
        1. Extrinsic calibration requires all cameras to have synchronized video
        2. Intrinsic videos may be added incrementally, but extrinsic must be complete
        3. Reconstruction uses the extrinsic-calibrated camera set

        Returns:
            Sorted list of camera IDs found in extrinsic directory.
            Empty list if directory doesn't exist or has no videos.
        """
        return self.get_cam_ids_in_dir(self.extrinsic_dir)

    def get_camera_count(self) -> int:
        """Return camera count derived from extrinsic directory.

        Returns:
            Number of cameras (cam_*.mp4 files in extrinsic directory).
        """
        return len(self.get_cam_ids())

    def all_instrinsic_mp4s_available(self) -> bool:
        """Check if intrinsic videos exist for every camera in the extrinsic set.

        The extrinsic directory defines the camera set. This checks that the
        intrinsic directory has a matching cam_N.mp4 for each extrinsic camera.
        """
        expected_cam_ids = self.get_cam_ids()
        if not expected_cam_ids:
            return False
        return self.missing_files_in_dir(self.intrinsic_dir, expected_cam_ids) == "NONE"

    def all_extrinsic_mp4s_available(self) -> bool:
        """Check if all extrinsic videos are present.

        Since the extrinsic directory is self-referential (it defines the camera
        set), this just checks that at least one camera exists.
        """
        return len(self.get_cam_ids()) > 0

    def missing_files_in_dir(self, directory: Path, expected_cam_ids: list[int]) -> str:
        """Return comma-separated list of missing cam_N.mp4 files.

        Compares actual files in directory against the expected camera ID set
        (typically derived from extrinsic directory).

        Args:
            directory: Path to check for files
            expected_cam_ids: Camera IDs that should have corresponding videos

        Returns:
            Comma-separated list like "cam_1.mp4,cam_3.mp4" or "NONE"
        """
        if not directory.exists():
            return ",".join([f"cam_{c}.mp4" for c in sorted(expected_cam_ids)])

        current_cam_ids = set(self.get_cam_ids_in_dir(directory))
        missing_cam_ids = sorted(set(expected_cam_ids) - current_cam_ids)

        if not missing_cam_ids:
            return "NONE"

        return ",".join([f"cam_{cam_id}.mp4" for cam_id in missing_cam_ids])

    def uncalibrated_cameras(self, camera_array: CameraArray) -> str:
        """
        Return comma-separated list of cameras lacking intrinsic calibration.

        Args:
            camera_array: Current camera array from Controller

        Returns:
            Comma-separated camera IDs or "NONE"
        """
        if not camera_array.cameras:
            return "NONE"

        uncalibrated = [
            str(cam.cam_id)
            for cam in camera_array.cameras.values()
            if cam.distortions is None and cam.matrix is None and cam.error is None
        ]

        return ",".join(uncalibrated) if uncalibrated else "NONE"

    def intrinsic_calibration_status(self, camera_array: CameraArray) -> str:
        """Return status of intrinsic calibration: COMPLETE or INCOMPLETE."""
        if camera_array.all_intrinsics_calibrated() and self.all_instrinsic_mp4s_available():
            return "COMPLETE"
        return "INCOMPLETE"

    def extrinsic_calibration_status(self, camera_array: CameraArray) -> str:
        """Return status of extrinsic calibration: COMPLETE or INCOMPLETE."""
        if camera_array.all_extrinsics_calibrated() and self.all_extrinsic_mp4s_available():
            return "COMPLETE"
        return "INCOMPLETE"

    def valid_recording_dirs(self) -> list[str]:
        """Return recording sessions ready for the extrinsic camera set."""
        return self.ready_recording_dirs(self.get_cam_ids())

    def valid_recording_dir_text(self) -> str:
        """Return comma-separated list of valid recording directories."""
        recording_dirs = self.valid_recording_dirs()
        return ",".join(recording_dirs) if recording_dirs else "NONE"

    def get_html_summary(self, camera_array: CameraArray) -> str:
        """Provide granular summary of calibration process state.

        Args:
            camera_array: Current camera array from Controller (source of truth)

        Returns:
            HTML string summarizing workspace state
        """
        expected_cam_ids = self.get_cam_ids()
        camera_count = len(expected_cam_ids)
        html = f"""
            <html>
                <head>
                    <style>
                        p {{
                            text-indent: 30px;
                        }}
                    </style>
                </head>
                <body>
                    <h4>Summary</h4>
                    <p>    Directory: {self.workspace_dir}</p>
                    <p>    Camera Count: {camera_count}</p>
                    <h4>Intrinsic Calibration: {self.intrinsic_calibration_status(camera_array)}</h4>
                    <p>    subdirectory: {self.intrinsic_dir}</p>
                    <p>    missing files: {self.missing_files_in_dir(self.intrinsic_dir, expected_cam_ids)}</p>
                    <p>    cameras needing calibration: {self.uncalibrated_cameras(camera_array)}</p>
                    <h4>Extrinsic Calibration: {self.extrinsic_calibration_status(camera_array)}</h4>
                    <p>    subdirectory: {self.extrinsic_dir}</p>
                    <p>    missing files: {self.missing_files_in_dir(self.extrinsic_dir, expected_cam_ids)}</p>
                    <h4>Recordings</h4>
                    <p>    valid directories: {self.valid_recording_dir_text()}</p>
                </body>
            </html>
            """
        return html
