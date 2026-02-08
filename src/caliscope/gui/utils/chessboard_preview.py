"""Shared chessboard preview rendering utility.

Used by both ProjectSetupView and CamerasTabWidget to render
chessboard patterns via QPainter without coupling the domain
model to Qt.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap

from caliscope.core.chessboard import Chessboard


def render_chessboard_pixmap(chessboard: Chessboard, size: int) -> QPixmap:
    """Render chessboard as alternating black/white squares via QPainter.

    The grid has (columns+1) x (rows+1) squares for a board with
    columns x rows internal corners. Maintains aspect ratio within
    the target size.
    """
    n_cols = chessboard.columns + 1
    n_rows = chessboard.rows + 1

    aspect = n_cols / n_rows
    if aspect > 1:
        width = size
        height = int(size / aspect)
    else:
        height = size
        width = int(size * aspect)

    square_w = width / n_cols
    square_h = height / n_rows

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.white)

    painter = QPainter(pixmap)
    for row in range(n_rows):
        for col in range(n_cols):
            if (row + col) % 2 == 1:
                painter.fillRect(
                    int(col * square_w),
                    int(row * square_h),
                    int(square_w + 0.5),
                    int(square_h + 0.5),
                    Qt.GlobalColor.black,
                )
    painter.end()
    return pixmap
