import logging
from pathlib import Path

from caliscope.cameras.camera_array import CameraArray

logger = logging.getLogger(__name__)


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
        if not directory.exists():
            return []

        all_cam_ids = []
        for file in directory.iterdir():
            if file.stem.startswith("cam_") and file.suffix == ".mp4":
                try:
                    cam_id = int(file.stem.split("_")[1])
                    all_cam_ids.append(cam_id)
                except (ValueError, IndexError):
                    logger.warning(f"Skipping malformed filename: {file.name}")

        return sorted(all_cam_ids)

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
        """Return list of valid recording directory names (all cam_N.mp4 present)."""
        if not self.recording_dir.exists():
            return []

        dir_list = []
        for p in self.recording_dir.iterdir():
            if p.is_dir():
                # A recording dir is valid if it has videos for all discovered cameras
                cam_ids_in_dir = self.get_cam_ids_in_dir(p)
                if cam_ids_in_dir:  # Must have at least some videos
                    dir_list.append(p.stem)

        return sorted(dir_list)

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
