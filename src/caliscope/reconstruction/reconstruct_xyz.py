"""Stage 2 of reconstruction: triangulate 2D landmarks into 3D trajectories.

Pure use case. Takes the in-memory ImagePoints produced by
process_synchronized_recording and writes the xyz csv, labelled csv, and .trc to the
tracker output directory. No streaming, no file read-back.
"""

import logging
from pathlib import Path

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints
from caliscope.export import xyz_to_trc, xyz_to_wide_labelled
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


def reconstruct_xyz(
    image_points: ImagePoints,
    camera_array: CameraArray,
    tracker: Tracker,
    output_dir: Path,
    xy_gap_fill: int = 3,
) -> None:
    """Triangulate image points and write xyz csv / labelled csv / trc to output_dir.

    Writes nothing when there are no 2D points or nothing triangulates -- a no-points
    run must not leave an empty xyz file (that would flip the reconstruction tab to a
    false COMPLETE). Filenames use the tracker name: xyz_{tracker.name}.{csv,trc}.
    """
    if image_points.df.empty:
        logger.warning("No 2D points to triangulate; skipping reconstruction output.")
        return

    filled_xy = image_points.fill_gaps(max_gap_size=xy_gap_fill)
    xyz_data = filled_xy.triangulate(camera_array)

    if xyz_data.df.empty:
        logger.warning("No points were triangulated; skipping reconstruction output.")
        return

    xyz_data.df.to_csv(output_dir / f"xyz_{tracker.name}.csv", index=False)

    labelled = xyz_to_wide_labelled(xyz_data.df, tracker)
    labelled.to_csv(output_dir / f"xyz_{tracker.name}_labelled.csv", index=False)

    xyz_to_trc(xyz_data.df, tracker=tracker, target_path=output_dir / f"xyz_{tracker.name}.trc")
