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

    Layout: Vertical stack of rows
    - Row 1: Board Shape: [rows] x [cols]
    - Row 2: Board Size: [width] x [height] [units]
    - Row 3: Invert checkbox
    - Row 4: Printed Edge: [value] cm

    This widget does NOT contain:
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
        """Build the widget layout with vertically stacked rows."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)  # We'll use explicit spacing between rows

        # Row 1: Board Shape: [rows] x [cols]
        shape_row = QHBoxLayout()
        shape_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        shape_row.addWidget(QLabel("Board Shape:"))

        self._row_spin = QSpinBox()
        self._row_spin.setValue(self._charuco_params["rows"])
        setup_spinbox_sizing(self._row_spin, min_value=4, max_value=999, padding=10)
        shape_row.addWidget(self._row_spin)

        shape_row.addWidget(QLabel("x"))

        self._column_spin = QSpinBox()
        self._column_spin.setValue(self._charuco_params["columns"])
        setup_spinbox_sizing(self._column_spin, min_value=3, max_value=999, padding=10)
        shape_row.addWidget(self._column_spin)

        shape_row.addStretch()
        main_layout.addLayout(shape_row)

        # 12px spacing between rows (style guide: row-to-row spacing)
        main_layout.addSpacing(12)

        # Row 2: Board Size: [width] x [height] [units]
        size_row = QHBoxLayout()
        size_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        size_row.addWidget(QLabel("Board Size:"))

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setValue(self._charuco_params["board_width"])
        setup_spinbox_sizing(self._width_spin, min_value=1, max_value=10000, padding=10)
        size_row.addWidget(self._width_spin)

        size_row.addWidget(QLabel("x"))

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setValue(self._charuco_params["board_height"])
        setup_spinbox_sizing(self._height_spin, min_value=1, max_value=10000, padding=10)
        size_row.addWidget(self._height_spin)

        self._units_combo = QComboBox()
        self._units_combo.addItems(["cm", "inch"])
        self._units_combo.setCurrentText(self._charuco_params["units"])
        size_row.addWidget(self._units_combo)

        size_row.addStretch()
        main_layout.addLayout(size_row)

        main_layout.addSpacing(12)

        # Row 3: Invert checkbox
        invert_row = QHBoxLayout()
        invert_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._invert_checkbox = QCheckBox("&Invert")
        self._invert_checkbox.setChecked(self._charuco_params["inverted"])
        invert_row.addWidget(self._invert_checkbox)

        invert_row.addStretch()
        main_layout.addLayout(invert_row)

        # Flexible stretch - allows top controls to separate from bottom
        main_layout.addStretch()

        # Row 4: Printed Edge: [value] cm (stays grouped with helper text below)
        edge_row = QHBoxLayout()
        edge_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        edge_row.addWidget(QLabel("Printed Edge:"))

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

        edge_row.addWidget(self._printed_edge_spin)
        edge_row.addWidget(QLabel("cm"))

        edge_row.addStretch()
        main_layout.addLayout(edge_row)

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
