"""ArUco marker set summary panel for extrinsic calibration.

Read-only summary of the loaded marker set. The TOML file is the interface
for editing — this panel shows what was loaded and provides reload/save/open actions.
"""

import logging
from pathlib import Path

import cv2
from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from caliscope.core.aruco_marker import ArucoMarkerSet

logger = logging.getLogger(__name__)


class ArucoMarkerSetPanel(QWidget):
    """Read-only summary panel for ArUco marker set configuration.

    Shows marker count, IDs, and sizes. Provides buttons to open the
    TOML directory, reload the config, and save all marker PNGs.

    Emits `config_changed` after a reload so callers can refresh state.
    """

    config_changed = Signal()

    def __init__(
        self,
        marker_set: ArucoMarkerSet,
        targets_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._marker_set = marker_set
        self._targets_dir = targets_dir
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Summary label
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._update_summary()
        layout.addWidget(self._summary_label)

        # Help link
        help_label = QLabel(
            '<a href="https://mprib.github.io/caliscope/aruco_calibration_set/">'
            "How to set up multi-marker calibration</a>"
        )
        help_label.setOpenExternalLinks(True)
        layout.addWidget(help_label)

        # Button row
        btn_row = QHBoxLayout()

        open_btn = QPushButton("Edit TOML")
        open_btn.setToolTip("Open aruco_marker_set.toml in your text editor")
        open_btn.clicked.connect(self._open_toml)
        btn_row.addWidget(open_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.setToolTip("Re-read aruco_marker_set.toml from disk")
        reload_btn.clicked.connect(self._reload)
        btn_row.addWidget(reload_btn)

        save_btn = QPushButton("Save All PNGs")
        save_btn.setToolTip("Save one PNG per marker to marker_images/ subfolder")
        save_btn.clicked.connect(self._save_all_pngs)
        btn_row.addWidget(save_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()

    def _update_summary(self) -> None:
        ms = self._marker_set
        lines = [f"<b>{len(ms.markers)} marker(s)</b>"]
        for mid in sorted(ms.markers.keys()):
            size_cm = ms.markers[mid].size_m * 100
            lines.append(f"  ID {mid}: {size_cm:.1f} cm")
        self._summary_label.setText("<br>".join(lines))

    def _open_toml(self) -> None:
        toml_path = self._targets_dir / "aruco_marker_set.toml"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(toml_path)))

    def _reload(self) -> None:
        toml_path = self._targets_dir / "aruco_marker_set.toml"
        if toml_path.exists():
            try:
                self._marker_set = ArucoMarkerSet.from_toml(toml_path)
                self._update_summary()
                self.config_changed.emit()
                logger.info("Reloaded ArUco marker set from %s", toml_path)
            except Exception as e:
                logger.error("Failed to reload marker set: %s", e)
        else:
            logger.warning("No aruco_marker_set.toml found at %s", toml_path)

    def _save_all_pngs(self) -> None:
        ms = self._marker_set
        out_dir = self._targets_dir / "marker_images"
        out_dir.mkdir(exist_ok=True)
        for mid, marker in ms.markers.items():
            pixel_size = int(marker.size_m * 8000)
            bgr = ms.generate_marker_image(mid, pixel_size)
            out_path = out_dir / f"marker_{mid}.png"
            cv2.imwrite(str(out_path), bgr)
            logger.info("Saved %s", out_path)

    @property
    def marker_set(self) -> ArucoMarkerSet:
        return self._marker_set
