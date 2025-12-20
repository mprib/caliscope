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
        self.intrinsic_dir = Path(workspace_dir, "calibration", "intrinsic")
        self.extrinsic_dir = Path(workspace_dir, "calibration", "extrinsic")
        self.recording_dir = Path(workspace_dir, "recordings")

    def get_ports_in_dir(self, directory: Path) -> list[int]:
        """
        Return list of port indices from video files in directory.

        Args:
            directory: Path to scan for port_N.mp4 files

        Returns:
            Sorted list of integer port numbers found
        """
        if not directory.exists():
            return []

        all_ports = []
        for file in directory.iterdir():
            if file.stem.startswith("port_") and file.suffix == ".mp4":
                try:
                    port = int(file.stem.split("_")[1])
                    all_ports.append(port)
                except (ValueError, IndexError):
                    logger.warning(f"Skipping malformed filename: {file.name}")

        return sorted(all_ports)

    def all_instrinsic_mp4s_available(self, camera_count: int) -> bool:
        """Check if all intrinsic videos are present for configured camera count."""
        return self.missing_files_in_dir(self.intrinsic_dir, camera_count) == "NONE"

    def all_extrinsic_mp4s_available(self, camera_count: int) -> bool:
        """Check if all extrinsic videos are present for configured camera count."""
        return self.missing_files_in_dir(self.extrinsic_dir, camera_count) == "NONE"

    def missing_files_in_dir(self, directory: Path, camera_count: int) -> str:
        """
        Return comma-separated list of missing port_N.mp4 files.

        Args:
            directory: Path to check for files
            camera_count: Expected number of cameras (ports 1..camera_count)

        Returns:
            Comma-separated list like "port_1.mp4,port_3.mp4" or "NONE"
        """
        if not directory.exists():
            return ",".join([f"port_{i}.mp4" for i in range(1, camera_count + 1)])

        target_ports = set(range(1, camera_count + 1))
        current_ports = set(self.get_ports_in_dir(directory))
        missing_ports = sorted(target_ports - current_ports)

        if not missing_ports:
            return "NONE"

        return ",".join([f"port_{port}.mp4" for port in missing_ports])

    def uncalibrated_cameras(self, camera_array: CameraArray) -> str:
        """
        Return comma-separated list of cameras lacking intrinsic calibration.

        Args:
            camera_array: Current camera array from Controller

        Returns:
            Comma-separated port numbers or "NONE"
        """
        if not camera_array.cameras:
            return "NONE"

        uncalibrated = [
            str(cam.port)
            for cam in camera_array.cameras.values()
            if cam.distortions is None and cam.matrix is None and cam.error is None
        ]

        return ",".join(uncalibrated) if uncalibrated else "NONE"

    def intrinsic_calibration_status(self, camera_array: CameraArray, camera_count: int) -> str:
        """Return status of intrinsic calibration: COMPLETE or INCOMPLETE."""
        if camera_array.all_intrinsics_calibrated() and self.all_instrinsic_mp4s_available(camera_count):
            return "COMPLETE"
        return "INCOMPLETE"

    def extrinsic_calibration_status(self, camera_array: CameraArray, camera_count: int) -> str:
        """Return status of extrinsic calibration: COMPLETE or INCOMPLETE."""
        if camera_array.all_extrinsics_calibrated() and self.all_extrinsic_mp4s_available(camera_count):
            return "COMPLETE"
        return "INCOMPLETE"

    def valid_recording_dirs(self) -> list[str]:
        """Return list of valid recording directory names (all port_N.mp4 present)."""
        if not self.recording_dir.exists():
            return []

        dir_list = []
        for p in self.recording_dir.iterdir():
            if p.is_dir():
                # A recording dir is valid if it has videos for all discovered ports
                ports_in_dir = self.get_ports_in_dir(p)
                if ports_in_dir:  # Must have at least some videos
                    dir_list.append(p.stem)

        return sorted(dir_list)

    def valid_recording_dir_text(self) -> str:
        """Return comma-separated list of valid recording directories."""
        recording_dirs = self.valid_recording_dirs()
        return ",".join(recording_dirs) if recording_dirs else "NONE"

    def get_html_summary(self, camera_array: CameraArray, camera_count: int) -> str:
        """
        Provide granular summary of calibration process state.

        Args:
            camera_array: Current camera array from Controller (source of truth)
            camera_count: Current camera count from Controller

        Returns:
            HTML string summarizing workspace state
        """
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
                    <h4>Intrinsic Calibration: {self.intrinsic_calibration_status(camera_array, camera_count)}</h4>
                    <p>    subdirectory: {self.intrinsic_dir}</p>
                    <p>    missing files: {self.missing_files_in_dir(self.intrinsic_dir, camera_count)}</p>
                    <p>    cameras needing calibration: {self.uncalibrated_cameras(camera_array)}</p>
                    <h4>Extrinsic Calibration: {self.extrinsic_calibration_status(camera_array, camera_count)}</h4>
                    <p>    subdirectory: {self.extrinsic_dir}</p>
                    <p>    missing files: {self.missing_files_in_dir(self.extrinsic_dir, camera_count)}</p>
                    <h4>Recordings</h4>
                    <p>    valid directories: {self.valid_recording_dir_text()}</p>
                </body>
            </html>
            """
        return html
