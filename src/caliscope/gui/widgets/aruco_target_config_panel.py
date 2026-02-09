"""ArUco target configuration panel for extrinsic calibration.

Allows user to configure dictionary, marker ID, and physical size for
the ArUco marker used in extrinsic calibration. Physical size is essential
for setting the world scale gauge.
"""

import logging

import cv2
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from caliscope.core.aruco_target import ArucoTarget
from caliscope.gui.utils.spinbox_utils import setup_spinbox_sizing

logger = logging.getLogger(__name__)


# Dictionary options: display name -> cv2 constant value
ARUCO_DICTIONARIES = [
    ("4x4 (50 markers)", cv2.aruco.DICT_4X4_50),
    ("4x4 (100 markers)", cv2.aruco.DICT_4X4_100),
    ("4x4 (250 markers)", cv2.aruco.DICT_4X4_250),
    ("5x5 (50 markers)", cv2.aruco.DICT_5X5_50),
    ("5x5 (100 markers)", cv2.aruco.DICT_5X5_100),
]


class ArucoTargetConfigPanel(QWidget):
    """ArUco target configuration for extrinsic calibration.

    Layout:
    - Row 1: Dictionary: [combo box]
    - Row 2: Marker ID: [spinbox]
    - Row 3: Marker Size: [spinbox] cm
    - Helper text explaining scale gauge

    Emits `config_changed` whenever any value changes.
    Use `get_aruco_target()` to build an ArucoTarget from current values.
    """

    config_changed = Signal()

    def __init__(
        self,
        initial_target: ArucoTarget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._setup_ui(initial_target)

    def _setup_ui(self, initial_target: ArucoTarget) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Row 1: Dictionary
        dict_row = QHBoxLayout()
        dict_row.addWidget(QLabel("Dictionary:"))
        self._dict_combo = QComboBox()
        for display_name, value in ARUCO_DICTIONARIES:
            self._dict_combo.addItem(display_name, value)
        # Set initial selection
        for i, (_, value) in enumerate(ARUCO_DICTIONARIES):
            if value == initial_target.dictionary:
                self._dict_combo.setCurrentIndex(i)
                break
        dict_row.addWidget(self._dict_combo)
        dict_row.addStretch()
        layout.addLayout(dict_row)

        # Row 2: Marker ID
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Marker ID:"))
        self._id_spin = QSpinBox()
        self._id_spin.setMinimum(0)
        self._id_spin.setMaximum(249)  # Max for 4x4_250
        initial_id = initial_target.marker_ids[0] if initial_target.marker_ids else 0
        self._id_spin.setValue(initial_id)
        setup_spinbox_sizing(self._id_spin, min_value=0, max_value=249)
        id_row.addWidget(self._id_spin)
        id_row.addStretch()
        layout.addLayout(id_row)

        # Row 3: Marker Size (in cm, domain uses meters)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Marker Size:"))
        self._size_spin = QDoubleSpinBox()
        self._size_spin.setDecimals(1)
        self._size_spin.setSingleStep(0.5)
        self._size_spin.setMinimum(0.5)
        self._size_spin.setMaximum(100.0)
        self._size_spin.setSuffix(" cm")
        # Convert meters to cm for display
        initial_cm = initial_target.marker_size_m * 100
        self._size_spin.setValue(initial_cm)
        setup_spinbox_sizing(self._size_spin, min_value=0.5, max_value=100.0)
        size_row.addWidget(self._size_spin)
        size_row.addStretch()
        layout.addLayout(size_row)

        # Helper text
        helper = QLabel("(Physical size sets the world scale gauge)")
        helper.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        layout.addWidget(helper)

        layout.addStretch()

        # Connect signals
        self._dict_combo.currentIndexChanged.connect(self._on_config_changed)
        self._id_spin.valueChanged.connect(self._on_config_changed)
        self._size_spin.valueChanged.connect(self._on_config_changed)

    def _on_config_changed(self) -> None:
        self.config_changed.emit()

    def get_aruco_target(self) -> ArucoTarget:
        """Build ArucoTarget from current widget values."""
        dictionary = self._dict_combo.currentData()
        marker_id = self._id_spin.value()
        marker_size_m = self._size_spin.value() / 100.0  # cm -> meters

        return ArucoTarget.single_marker(
            marker_id=marker_id,
            marker_size_m=marker_size_m,
            dictionary=dictionary,
        )
