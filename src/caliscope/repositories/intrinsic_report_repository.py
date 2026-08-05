"""Repository for intrinsic calibration reports.

Stores quality metrics and frame selection data for intrinsic calibration.
Reports are stored per-camera in TOML format at:
    calibration/intrinsic/reports/cam_{N}.toml
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

    def _cam_path(self, cam_id: int) -> Path:
        """Get file path for a specific camera's report."""
        return self._reports_dir / f"cam_{cam_id}.toml"

    def save(self, cam_id: int, report: IntrinsicCalibrationReport) -> None:
        """Save calibration report for a camera.

        Args:
            cam_id: Camera identifier
            report: Calibration report to save

        Raises:
            ValueError: If save operation fails
        """
        self._reports_dir.mkdir(parents=True, exist_ok=True)

        # Explicitly convert to Python native types - numpy types can't be serialized by rtoml
        data = {
            "rmse": float(report.rmse),
            "frames_used": int(report.frames_used),
            "coverage_fraction": float(report.coverage_fraction),
            "edge_coverage_fraction": float(report.edge_coverage_fraction),
            "corner_coverage_fraction": float(report.corner_coverage_fraction),
            "orientation_sufficient": bool(report.orientation_sufficient),
            "orientation_count": int(report.orientation_count),
            "selected_frames": [int(f) for f in report.selected_frames],
        }

        path = self._cam_path(cam_id)
        try:
            with open(path, "w") as f:
                rtoml.dump(data, f)
            logger.info(f"Saved intrinsic report for cam_id {cam_id}: RMSE={report.rmse:.3f}px")
        except Exception as e:
            raise ValueError(f"Failed to save intrinsic report for cam_id {cam_id}: {e}") from e

    def load(self, cam_id: int) -> IntrinsicCalibrationReport | None:
        """Load calibration report for a camera.

        Args:
            cam_id: Camera identifier

        Returns:
            Calibration report or None if not found

        Raises:
            ValueError: If file exists but contains invalid data
        """
        path = self._cam_path(cam_id)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = rtoml.load(f)

            return IntrinsicCalibrationReport(
                rmse=float(data["rmse"]),
                frames_used=int(data["frames_used"]),
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
            Dictionary mapping camera identifiers to reports
        """
        reports: dict[int, IntrinsicCalibrationReport] = {}

        if not self._reports_dir.exists():
            return reports

        for path in self._reports_dir.glob("cam_*.toml"):
            try:
                cam_id = int(path.stem.split("_")[1])
                report = self.load(cam_id)
                if report is not None:
                    reports[cam_id] = report
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid report file {path}: {e}")

        return reports

    def delete(self, cam_id: int) -> bool:
        """Delete report for a camera.

        Args:
            cam_id: Camera identifier

        Returns:
            True if file was deleted, False if it didn't exist
        """
        path = self._cam_path(cam_id)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted intrinsic report for cam_id {cam_id}")
            return True
        return False
