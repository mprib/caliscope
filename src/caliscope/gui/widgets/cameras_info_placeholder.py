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

<p>This tab calibrates each camera's lens (focal length, distortion) from videos in
<code>calibration/intrinsic/</code>. This project has none — and that can be fine.
Extrinsic calibration can recover lens parameters on its own if the capture supports it:</p>

<ul>
<li>Move the target toward and away from the cameras, not just across the view.</li>
<li>Measure marker sizes accurately — they set the world scale.</li>
<li>No fisheye lenses. Those need intrinsic calibration first.</li>
</ul>

<p>If that matches your capture, continue on the <b>Calibrate</b> tab. To calibrate
intrinsics here instead, add <code>calibration/intrinsic/cam_N.mp4</code> videos and
this tab will activate.</p>

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
