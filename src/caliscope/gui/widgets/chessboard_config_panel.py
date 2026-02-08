"""Reusable chessboard pattern configuration widget.

Used in intrinsic calibration workflow for configuring chessboard patterns.
Simpler than charuco — only rows and columns (internal corners).
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from caliscope.core.chessboard import Chessboard
from caliscope.gui.theme import Typography
from caliscope.gui.utils.spinbox_utils import setup_spinbox_sizing

logger = logging.getLogger(__name__)


class ChessboardConfigPanel(QWidget):
    """Reusable chessboard pattern configuration widget.

    Emits `config_changed` whenever any configuration value is modified.
    Use `get_chessboard()` to build a Chessboard instance from current values.

    Layout: Vertical stack with shape row and helper text
    - Row 1: Board Shape: [columns] x [rows] corners
    - Helper text: Dynamic square count explanation

    This widget does NOT contain:
    - Board preview image (responsibility of the parent view)
    - Physical size configuration (chessboard uses unit spacing)
    """

    config_changed = Signal()

    def __init__(
        self,
        initial_chessboard: Chessboard,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the panel with values from an existing Chessboard.

        Args:
            initial_chessboard: Chessboard instance to populate initial widget values
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._initial_rows = initial_chessboard.rows
        self._initial_columns = initial_chessboard.columns
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Build the widget layout with shape row and helper text."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Row 1: Board Shape: [columns] x [rows] corners
        shape_row = QHBoxLayout()
        shape_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        shape_row.addWidget(QLabel("Board Shape:"))

        self._column_spin = QSpinBox()
        self._column_spin.setValue(self._initial_columns)
        setup_spinbox_sizing(self._column_spin, min_value=3, max_value=99, padding=10)
        shape_row.addWidget(self._column_spin)

        shape_row.addWidget(QLabel("x"))

        self._row_spin = QSpinBox()
        self._row_spin.setValue(self._initial_rows)
        setup_spinbox_sizing(self._row_spin, min_value=3, max_value=99, padding=10)
        shape_row.addWidget(self._row_spin)

        shape_row.addWidget(QLabel("corners"))

        shape_row.addStretch()
        main_layout.addLayout(shape_row)

        # Helper text explaining square counts (muted italic, dynamically updated)
        self._helper_label = QLabel()
        self._helper_label.setStyleSheet(Typography.HELPER_TEXT)
        self._update_helper_text()
        main_layout.addWidget(self._helper_label)

        main_layout.addStretch()

    def _connect_signals(self) -> None:
        """Connect widget signals to emit config_changed."""
        self._row_spin.valueChanged.connect(self._on_config_changed)
        self._column_spin.valueChanged.connect(self._on_config_changed)

    def _on_config_changed(self) -> None:
        """Handle any configuration change."""
        self._update_helper_text()
        self.config_changed.emit()

    def _update_helper_text(self) -> None:
        """Update helper text with current square counts."""
        cols = self._column_spin.value()
        rows = self._row_spin.value()
        square_cols = cols + 1
        square_rows = rows + 1
        self._helper_label.setText(f"(internal corners — your grid has {square_cols} x {square_rows} squares)")

    def get_chessboard(self) -> Chessboard:
        """Build a Chessboard instance from current widget values.

        Returns:
            New Chessboard with configuration from this panel
        """
        return Chessboard(
            rows=self._row_spin.value(),
            columns=self._column_spin.value(),
        )
