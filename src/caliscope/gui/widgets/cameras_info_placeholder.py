"""Informational placeholder for the Cameras tab when intrinsic videos are absent.

Shown in place of CamerasTabWidget so the tab stays clickable and explains what
the workspace holds: intrinsics already loaded from camera_array.toml, or none
at all with the risks of estimating them during extrinsic calibration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from caliscope.gui.tab_names import TabName

if TYPE_CHECKING:
    from caliscope.cameras.camera_array import CameraArray, CameraData
    from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)

DOCS_URL = "https://mprib.github.io/caliscope/extrinsic_calibration_reference/#skipping-intrinsic-calibration"

_NO_INTRINSICS_HTML = f"""
<h3>No intrinsic calibration videos</h3>

<p>This tab calibrates each camera's lens (focal length, distortion) from videos in
<code>calibration/intrinsic/</code>. This project has none.</p>

<p>Without them, extrinsic calibration starts each camera from a rough guess (focal
length from resolution, zero distortion) and refines focal length and the first two
distortion terms during bundle adjustment. The principal point and remaining distortion
terms stay at their assumed values. This path is experimental: it has not been validated
on real-world data to the same degree as calibrating intrinsics first, and results may
be poor even when the solver reports success.</p>

<ul>
<li>Move the target toward and away from the cameras and out to the edges of each view,
not just across the middle.</li>
<li>Measure marker sizes accurately. They set the world scale.</li>
<li>Fisheye lenses need intrinsic calibration first; this path cannot recover them.</li>
</ul>

<p>Recording intrinsic videos is the reliable path. If you continue without them, judge
the result by the quality report on the <b>{TabName.EXTRINSICS}</b> tab rather than
assuming it worked.</p>

<p>To calibrate intrinsics here, add <code>calibration/intrinsic/cam_N.mp4</code> videos
and this tab will activate.</p>

<p><a href="{DOCS_URL}">Skipping intrinsic calibration: documentation</a></p>
"""


def _camera_row(camera: CameraData) -> str:
    assert camera.matrix is not None
    fx, fy = camera.matrix[0, 0], camera.matrix[1, 1]
    cx, cy = camera.matrix[0, 2], camera.matrix[1, 2]
    error = f"{camera.error:.3f}" if camera.error is not None else "n/a"
    model = "fisheye" if camera.fisheye else "standard"
    return (
        f"<tr><td>Cam {camera.cam_id}</td><td>{camera.size[0]}x{camera.size[1]}</td>"
        f"<td>{fx:.1f}, {fy:.1f}</td><td>{cx:.1f}, {cy:.1f}</td><td>{error}</td><td>{model}</td></tr>"
    )


def _loaded_intrinsics_html(calibrated: list[CameraData], uncalibrated: list[CameraData]) -> str:
    rows = "".join(_camera_row(camera) for camera in calibrated)
    html = f"""
<h3>Intrinsics loaded from camera_array.toml</h3>

<p>There are no videos in <code>calibration/intrinsic/</code>, so these values cannot be
reviewed or redone here. They came from an earlier calibration or an import.</p>

<table cellpadding="4">
<tr><th align="left">Camera</th><th align="left">Resolution</th><th align="left">Focal (px)</th>
<th align="left">Principal point</th><th align="left">RMSE</th><th align="left">Model</th></tr>
{rows}
</table>

<p>To recalibrate a camera, add <code>calibration/intrinsic/cam_N.mp4</code> for it and
this tab will activate. The new result replaces the stored values for that camera.</p>
"""
    if uncalibrated:
        ids = ", ".join(str(camera.cam_id) for camera in uncalibrated)
        html += f"""
<p><b>Cameras {ids} have no intrinsics.</b> Extrinsic calibration will have to estimate
them. This path is experimental and can give poor results; see the
<a href="{DOCS_URL}">documentation on skipping intrinsic calibration</a>.</p>
"""
    return html


def placeholder_html(camera_array: CameraArray) -> str:
    """Explain the intrinsic situation for the cameras in the workspace."""
    cameras = [camera_array.cameras[cam_id] for cam_id in sorted(camera_array.cameras)]
    calibrated = [camera for camera in cameras if camera.matrix is not None]
    uncalibrated = [camera for camera in cameras if camera.matrix is None]
    if calibrated:
        return _loaded_intrinsics_html(calibrated, uncalibrated)
    return _NO_INTRINSICS_HTML


class CamerasInfoPlaceholder(QWidget):
    """Explanatory widget shown when the Cameras tab has no videos to work with."""

    def __init__(self, coordinator: WorkspaceCoordinator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._coordinator = coordinator

        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setOpenExternalLinks(True)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self._label.setMaximumWidth(720)

        # Center the fixed-width column; row layout keeps heightForWidth intact
        # so the wrapped label gets its full height (alignment flags do not).
        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(24, 24, 24, 24)
        row.addStretch()
        row.addWidget(self._label)
        row.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        coordinator.status_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the camera array so the text follows the workspace."""
        self._label.setText(placeholder_html(self._coordinator.camera_array))
