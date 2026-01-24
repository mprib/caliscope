"""Read-only coverage heatmap widget for visualizing camera pair observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CoverageHeatmapWidget(QWidget):
    """Read-only heatmap showing shared observations between camera pairs.

    Displays an NxN matrix where:
    - Diagonal [i,i]: Total observations for camera i (blue gradient)
    - Off-diagonal [i,j]: Observations shared between cameras i and j (green gradient)
    """

    # Colors
    COLOR_KILLED = QColor(80, 80, 80)  # Dark gray for killed linkages
    COLOR_ZERO = QColor(40, 40, 40)  # Very dark for no observations
    COLOR_LOW = QColor(60, 120, 60)  # Green gradient for counts
    COLOR_HIGH = QColor(60, 200, 60)

    # Layout constants
    MARGIN = 30  # Space for row/column labels

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._coverage: NDArray[np.int64] | None = None
        self._killed_linkages: set[tuple[int, int]] = set()

        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(
        self,
        coverage: NDArray[np.int64],
        killed_linkages: set[tuple[int, int]],
    ) -> None:
        """Update heatmap data.

        Args:
            coverage: (N, N) matrix of observation counts
            killed_linkages: Set of (cam_a, cam_b) tuples that are killed
        """
        self._coverage = coverage
        self._killed_linkages = killed_linkages
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        """Draw the heatmap."""
        if self._coverage is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        n = len(self._coverage)
        if n == 0:
            return

        # Calculate cell size
        cell_size = self._cell_size()
        if cell_size <= 0:
            return

        # Find max value for color scaling (excluding diagonal)
        max_val = self._max_off_diagonal()

        # Draw cells
        font = QFont("Monospace", 9)
        painter.setFont(font)

        for i in range(n):
            for j in range(n):
                x = self.MARGIN + j * cell_size
                y = self.MARGIN + i * cell_size

                count = int(self._coverage[i, j])
                is_killed = self._is_linkage_killed(i, j)

                # Determine color
                color = self._cell_color(i, j, count, is_killed, max_val)

                painter.fillRect(x, y, cell_size - 1, cell_size - 1, color)

                # Draw count text (or X for killed)
                painter.setPen(Qt.GlobalColor.white)
                text = "X" if is_killed and i != j else str(count)
                painter.drawText(x, y, cell_size - 1, cell_size - 1, Qt.AlignmentFlag.AlignCenter, text)

        # Draw row/column labels
        painter.setPen(Qt.GlobalColor.white)
        for i in range(n):
            # Column headers
            painter.drawText(
                self.MARGIN + i * cell_size,
                0,
                cell_size,
                self.MARGIN,
                Qt.AlignmentFlag.AlignCenter,
                f"C{i}",
            )
            # Row headers
            painter.drawText(
                0,
                self.MARGIN + i * cell_size,
                self.MARGIN,
                cell_size,
                Qt.AlignmentFlag.AlignCenter,
                f"C{i}",
            )

    def _cell_size(self) -> int:
        """Calculate cell size based on widget dimensions and camera count."""
        if self._coverage is None:
            return 0

        n = len(self._coverage)
        if n == 0:
            return 0

        available_width = self.width() - self.MARGIN
        available_height = self.height() - self.MARGIN
        return min(available_width // n, available_height // n)

    def _max_off_diagonal(self) -> int:
        """Find maximum off-diagonal value for color scaling."""
        if self._coverage is None:
            return 1

        n = len(self._coverage)
        max_val = 1
        for i in range(n):
            for j in range(n):
                if i != j and self._coverage[i, j] > max_val:
                    max_val = int(self._coverage[i, j])
        return max_val

    def _cell_color(self, i: int, j: int, count: int, is_killed: bool, max_val: int) -> QColor:
        """Determine color for a cell based on its state."""
        if i == j:
            # Diagonal: blue gradient based on observation count
            intensity = min(count / max(max_val * 2, 1), 1.0)
            return QColor(60, 60, int(100 + 155 * intensity))
        elif is_killed:
            return self.COLOR_KILLED
        elif count == 0:
            return self.COLOR_ZERO
        else:
            # Off-diagonal: green gradient
            intensity = count / max_val
            r = int(self.COLOR_LOW.red() + (self.COLOR_HIGH.red() - self.COLOR_LOW.red()) * intensity)
            g = int(self.COLOR_LOW.green() + (self.COLOR_HIGH.green() - self.COLOR_LOW.green()) * intensity)
            b = int(self.COLOR_LOW.blue() + (self.COLOR_HIGH.blue() - self.COLOR_LOW.blue()) * intensity)
            return QColor(r, g, b)

    def _is_linkage_killed(self, i: int, j: int) -> bool:
        """Check if linkage between cameras i and j is killed."""
        normalized = (min(i, j), max(i, j))
        return normalized in self._killed_linkages
