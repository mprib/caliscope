"""Reusable charuco board configuration widget.

Extracted from CharucoWidget for embedding in ProjectSetupView and
potential future use in other tabs (Intrinsics, Landmarks).
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from caliscope.core.charuco import Charuco
from caliscope.gui.utils.spinbox_utils import setup_spinbox_sizing

logger = logging.getLogger(__name__)


class CharucoConfigPanel(QWidget):
    """Reusable charuco board configuration widget.

    Emits `config_changed` whenever any configuration value is modified.
    Use `get_charuco()` to build a Charuco instance from current values.

    This widget contains:
    - Row/column spinboxes
    - Board width/height spinboxes
    - Units dropdown (cm/inch)
    - Invert checkbox
    - Printed edge length spinbox (for "true up" functionality)

    It does NOT contain:
    - Board preview image (responsibility of the parent view)
    - PNG save buttons (responsibility of the parent view)
    """

    config_changed = Signal()

    def __init__(
        self,
        initial_charuco: Charuco,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the panel with values from an existing Charuco.

        Args:
            initial_charuco: Charuco instance to populate initial widget values
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._charuco_params = self._extract_params(initial_charuco)
        self._setup_ui()
        self._connect_signals()

    def _extract_params(self, charuco: Charuco) -> dict:
        """Extract configuration parameters from a Charuco instance."""
        return {
            "columns": charuco.columns,
            "rows": charuco.rows,
            "board_width": charuco.board_width,
            "board_height": charuco.board_height,
            "units": charuco.units,
            "inverted": charuco.inverted,
            "dictionary": charuco.dictionary,
            "aruco_scale": charuco.aruco_scale,
            "square_size_overide_cm": charuco.square_size_overide_cm,
            "legacy_pattern": charuco.legacy_pattern,
        }

    def _setup_ui(self) -> None:
        """Build the widget layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Board shape and size configuration ---
        config_layout = QHBoxLayout()
        config_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Shape group (row x col)
        shape_group = QGroupBox("row x col")
        shape_layout = QHBoxLayout(shape_group)
        shape_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._row_spin = QSpinBox()
        self._row_spin.setValue(self._charuco_params["rows"])
        setup_spinbox_sizing(self._row_spin, min_value=4, max_value=999, padding=10)

        self._column_spin = QSpinBox()
        self._column_spin.setValue(self._charuco_params["columns"])
        setup_spinbox_sizing(self._column_spin, min_value=3, max_value=999, padding=10)

        # Note: displayed as "row x col" but row_spin comes first visually
        shape_layout.addWidget(self._row_spin)
        shape_layout.addWidget(self._column_spin)
        config_layout.addWidget(shape_group)

        # Size group (target board dimensions)
        size_group = QGroupBox("Target Board Size")
        size_layout = QHBoxLayout(size_group)
        size_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setValue(self._charuco_params["board_width"])
        setup_spinbox_sizing(self._width_spin, min_value=1, max_value=10000, padding=10)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setValue(self._charuco_params["board_height"])
        setup_spinbox_sizing(self._height_spin, min_value=1, max_value=10000, padding=10)

        self._units_combo = QComboBox()
        self._units_combo.addItems(["cm", "inch"])
        self._units_combo.setCurrentText(self._charuco_params["units"])

        size_layout.addWidget(self._width_spin)
        size_layout.addWidget(self._height_spin)
        size_layout.addWidget(self._units_combo)
        config_layout.addWidget(size_group)

        # Invert checkbox
        self._invert_checkbox = QCheckBox("&Invert")
        self._invert_checkbox.setChecked(self._charuco_params["inverted"])
        config_layout.addWidget(self._invert_checkbox)

        main_layout.addLayout(config_layout)

        # --- Printed edge length (true up) ---
        edge_layout = QHBoxLayout()
        edge_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        edge_label = QLabel("Actual Printed Square Edge Length:")
        edge_layout.addWidget(edge_label)

        self._printed_edge_spin = QDoubleSpinBox()
        self._printed_edge_spin.setSingleStep(0.01)
        self._printed_edge_spin.setDecimals(2)
        self._printed_edge_spin.setMinimum(0.01)
        self._printed_edge_spin.setMaximum(1000.0)
        self._printed_edge_spin.setMaximumWidth(100)

        override_value = self._charuco_params["square_size_overide_cm"]
        if override_value is not None:
            self._printed_edge_spin.setValue(override_value)
        else:
            # Default to a reasonable value if not set
            self._printed_edge_spin.setValue(5.0)

        edge_layout.addWidget(self._printed_edge_spin)
        edge_layout.addWidget(QLabel("cm"))

        main_layout.addLayout(edge_layout)

    def _connect_signals(self) -> None:
        """Connect widget signals to emit config_changed."""
        self._row_spin.valueChanged.connect(self._on_config_changed)
        self._column_spin.valueChanged.connect(self._on_config_changed)
        self._width_spin.valueChanged.connect(self._on_config_changed)
        self._height_spin.valueChanged.connect(self._on_config_changed)
        self._units_combo.currentIndexChanged.connect(self._on_config_changed)
        self._invert_checkbox.stateChanged.connect(self._on_config_changed)
        self._printed_edge_spin.valueChanged.connect(self._on_config_changed)

    def _on_config_changed(self) -> None:
        """Handle any configuration change."""
        self.config_changed.emit()

    def get_charuco(self) -> Charuco:
        """Build a Charuco instance from current widget values.

        Returns:
            New Charuco with configuration from this panel
        """
        return Charuco(
            columns=self._column_spin.value(),
            rows=self._row_spin.value(),
            board_height=self._height_spin.value(),
            board_width=self._width_spin.value(),
            units=self._units_combo.currentText(),
            dictionary=self._charuco_params["dictionary"],
            aruco_scale=self._charuco_params["aruco_scale"],
            square_size_overide_cm=round(self._printed_edge_spin.value(), 2),
            inverted=self._invert_checkbox.isChecked(),
            legacy_pattern=self._charuco_params["legacy_pattern"],
        )

    def set_printed_edge_length(self, cm: float) -> None:
        """Update the printed edge length override.

        Args:
            cm: Edge length in centimeters
        """
        # Block signals to avoid triggering config_changed during programmatic update
        self._printed_edge_spin.blockSignals(True)
        self._printed_edge_spin.setValue(cm)
        self._printed_edge_spin.blockSignals(False)
