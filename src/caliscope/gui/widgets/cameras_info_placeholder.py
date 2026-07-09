"""Informational placeholder for the Cameras tab when intrinsic videos are absent.

Shown in place of CamerasTabWidget so the tab stays clickable and can explain
that skipping intrinsic calibration is a supported path with prerequisites,
rather than presenting a greyed-out tab that reads as "stuck".
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

DOCS_URL = "https://mprib.github.io/caliscope/extrinsic_calibration/#skipping-intrinsic-calibration"

_PLACEHOLDER_HTML = f"""
<h3>No intrinsic calibration videos</h3>

<p>This tab calibrates each camera's intrinsics (focal length, distortion) from
per-camera videos in <code>calibration/intrinsic/</code>, and this project does not
have one for every camera in the extrinsic set.</p>

<p><b>That can be intentional.</b> Extrinsic calibration can recover focal length
and distortion jointly with the camera poses — optional, if you capture for it:</p>

<ul>
<li><b>Sweep the target through depth.</b> Move it toward and away from the
cameras, not just across their views. Without depth variation, focal length
cannot be recovered and the calibration falls back to a rough guess.</li>
<li><b>Measure marker sizes accurately.</b> Marker size supplies both world scale
and the rigid geometry that holds the solve together. Static anchor markers
strengthen it further.</li>
<li><b>Use markers large enough</b> to detect reliably across the volume.</li>
<li><b>No fisheye cameras.</b> Fisheye lenses require intrinsic calibration here
first; an extrinsic-only calibration fails for them outright.</li>
</ul>

<p>If that matches your capture, continue on the <b>Calibrate</b> tab.
To calibrate intrinsics here instead, add per-camera videos as
<code>calibration/intrinsic/cam_N.mp4</code> and this tab will activate.</p>

<p><a href="{DOCS_URL}">Skipping intrinsic calibration — documentation</a></p>
"""


class CamerasInfoPlaceholder(QWidget):
    """Static explanatory widget shown when the Cameras tab has no videos to work with."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        label = QLabel(_PLACEHOLDER_HTML)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        label.setMaximumWidth(640)

        # Center the fixed-width column; row layout keeps heightForWidth intact
        # so the wrapped label gets its full height (alignment flags do not).
        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(24, 24, 24, 24)
        row.addStretch()
        row.addWidget(label)
        row.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
