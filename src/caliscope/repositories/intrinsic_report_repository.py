"""Repository for intrinsic calibration reports.

Stores quality metrics and frame selection data for intrinsic calibration.
Reports are stored per-camera in TOML format at:
    calibration/intrinsic/reports/port_{N}.toml
"""

import logging
from pathlib import Path

import rtoml

from caliscope.core.calibrate_intrinsics import IntrinsicCalibrationReport

logger = logging.getLogger(__name__)


class IntrinsicReportRepository:
    """Persistence gateway for intrinsic calibration reports.

    Each camera's report is stored in a separate TOML file, enabling
    independent updates and lazy loading. The reports directory is
    created automatically when saving the first report.
    """

    def __init__(self, reports_dir: Path) -> None:
        """Initialize repository.

        Args:
            reports_dir: Directory for report files (e.g., calibration/intrinsic/reports/)
        """
        self._reports_dir = reports_dir

    def _port_path(self, port: int) -> Path:
        """Get file path for a specific port's report."""
        return self._reports_dir / f"port_{port}.toml"

    def save(self, port: int, report: IntrinsicCalibrationReport) -> None:
        """Save calibration report for a camera.

        Args:
            port: Camera port number
            report: Calibration report to save

        Raises:
            ValueError: If save operation fails
        """
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        # Explicitly convert to Python native types - numpy types can't be serialized by rtoml
        data = {
            "in_sample_rmse": float(report.in_sample_rmse),
            "out_of_sample_rmse": float(report.out_of_sample_rmse),
            "frames_used": int(report.frames_used),
            "holdout_frame_count": int(report.holdout_frame_count),
            "coverage_fraction": float(report.coverage_fraction),
            "edge_coverage_fraction": float(report.edge_coverage_fraction),
            "corner_coverage_fraction": float(report.corner_coverage_fraction),
            "orientation_sufficient": bool(report.orientation_sufficient),
            "orientation_count": int(report.orientation_count),
            "selected_frames": [int(f) for f in report.selected_frames],
        }

        path = self._port_path(port)
        try:
            with open(path, "w") as f:
                rtoml.dump(data, f)
            logger.info(f"Saved intrinsic report for port {port}: RMSE={report.in_sample_rmse:.3f}px")
        except Exception as e:
            raise ValueError(f"Failed to save intrinsic report for port {port}: {e}") from e

    def load(self, port: int) -> IntrinsicCalibrationReport | None:
        """Load calibration report for a camera.

        Args:
            port: Camera port number

        Returns:
            Calibration report or None if not found

        Raises:
            ValueError: If file exists but contains invalid data
        """
        path = self._port_path(port)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = rtoml.load(f)

            return IntrinsicCalibrationReport(
                in_sample_rmse=float(data["in_sample_rmse"]),
                out_of_sample_rmse=float(data["out_of_sample_rmse"]),
                frames_used=int(data["frames_used"]),
                holdout_frame_count=int(data["holdout_frame_count"]),
                coverage_fraction=float(data["coverage_fraction"]),
                edge_coverage_fraction=float(data["edge_coverage_fraction"]),
                corner_coverage_fraction=float(data["corner_coverage_fraction"]),
                orientation_sufficient=bool(data["orientation_sufficient"]),
                orientation_count=int(data["orientation_count"]),
                selected_frames=tuple(int(f) for f in data["selected_frames"]),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in {path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load intrinsic report from {path}: {e}") from e

    def load_all(self) -> dict[int, IntrinsicCalibrationReport]:
        """Load all available reports.

        Returns:
            Dictionary mapping port numbers to reports
        """
        reports: dict[int, IntrinsicCalibrationReport] = {}

        if not self._reports_dir.exists():
            return reports

        for path in self._reports_dir.glob("port_*.toml"):
            try:
                port = int(path.stem.split("_")[1])
                report = self.load(port)
                if report is not None:
                    reports[port] = report
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid report file {path}: {e}")

        return reports

    def delete(self, port: int) -> bool:
        """Delete report for a camera.

        Args:
            port: Camera port number

        Returns:
            True if file was deleted, False if it didn't exist
        """
        path = self._port_path(port)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted intrinsic report for port {port}")
            return True
        return False
