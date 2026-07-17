"""Welcome screen shown when no project is loaded."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from caliscope.gui import ICONS_DIR
from caliscope.gui.theme import Colors, Styles
from caliscope.gui.widgets.link_label import LinkLabel

_MAX_RECENT = 8


class WelcomeWidget(QWidget):
    open_project_requested = Signal()
    recent_project_selected = Signal(str)

    def __init__(self, recent_projects: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recent_projects = recent_projects[:_MAX_RECENT]
        self._interactive_widgets: list[QWidget] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)

        outer.addStretch(1)

        column = QVBoxLayout()
        column.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        column.setSpacing(12)
        outer.addLayout(column)

        # --- App icon ---
        icon_widget = self._build_icon()
        column.addWidget(icon_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

        # --- Title ---
        title = QLabel("Caliscope")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window_text = self.palette().color(QPalette.ColorRole.WindowText).name()
        title.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {window_text};")
        column.addWidget(title)

        # --- Subtitle ---
        subtitle = QLabel("Multicamera Calibration for Motion Capture Workflows")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"font-size: 13px; color: {window_text};")
        column.addWidget(subtitle)

        column.addSpacing(16)

        # --- Primary button ---
        self._open_button = QPushButton("New / Open Project…")
        self._open_button.setStyleSheet(Styles.PRIMARY_BUTTON)
        self._open_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._open_button.clicked.connect(self.open_project_requested)
        self._interactive_widgets.append(self._open_button)
        column.addWidget(self._open_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        column.addSpacing(16)

        # --- Recent Projects ---
        self._recents_container = QWidget()
        recents_layout = QVBoxLayout(self._recents_container)
        recents_layout.setContentsMargins(0, 0, 0, 0)
        recents_layout.setSpacing(4)

        recents_header = QLabel("Recent Projects")
        recents_header.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {self.palette().color(QPalette.ColorRole.WindowText).name()};"
        )
        recents_layout.addWidget(recents_header)

        self._recent_links: list[LinkLabel] = []
        for project_path in self._recent_projects:
            p = Path(project_path)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)

            link = LinkLabel(font_size_px=13)
            link.setText(p.name)
            link.setToolTip(str(p))
            path_str = project_path
            link.clicked.connect(lambda checked=False, s=path_str: self.recent_project_selected.emit(s))
            self._interactive_widgets.append(link)
            self._recent_links.append(link)
            row.addWidget(link)

            parent_display = str(p.parent).replace(str(Path.home()), "~")
            dimmed = QLabel(parent_display)
            placeholder_color = self.palette().color(QPalette.ColorRole.PlaceholderText).name()
            dimmed.setStyleSheet(f"font-size: 13px; color: {placeholder_color};")
            row.addWidget(dimmed)
            row.addStretch()

            recents_layout.addLayout(row)

        self._recents_container.setVisible(len(self._recent_projects) > 0)
        column.addWidget(self._recents_container, alignment=Qt.AlignmentFlag.AlignHCenter)

        column.addSpacing(12)

        # --- Status line ---
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setMaximumWidth(400)
        self._status_label.hide()
        column.addWidget(self._status_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # --- Progress bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setMaximumWidth(300)
        self._progress_bar.hide()
        column.addWidget(self._progress_bar, alignment=Qt.AlignmentFlag.AlignHCenter)

        # --- Helper text (first-time users only) ---
        if not self._recent_projects:
            helper = QLabel(
                "A project is a folder. Choose an empty folder to start a new project, "
                "or a previous project folder to reopen it."
            )
            helper.setAlignment(Qt.AlignmentFlag.AlignCenter)
            helper.setWordWrap(True)
            helper.setMaximumWidth(350)
            helper.setStyleSheet(f"font-size: 11px; color: {window_text};")
            column.addWidget(helper, alignment=Qt.AlignmentFlag.AlignHCenter)

        outer.addStretch(1)

        # --- Tab order ---
        self.setTabOrder(self._open_button, self._recent_links[0] if self._recent_links else self._open_button)
        for i in range(len(self._recent_links) - 1):
            self.setTabOrder(self._recent_links[i], self._recent_links[i + 1])

        self._open_button.setFocus()

    def _build_icon(self) -> QSvgWidget:
        svg_path = ICONS_DIR / "box3d-center.svg"
        svg_text = svg_path.read_text()
        text_color = self.palette().color(QPalette.ColorRole.WindowText).name()
        svg_text = svg_text.replace("#000000", text_color)
        icon = QSvgWidget()
        icon.setFixedSize(96, 96)
        icon.load(QByteArray(svg_text.encode()))
        return icon

    def set_loading(self, project_path: str) -> None:
        folder_name = Path(project_path).name
        self._status_label.setText(f"Loading {folder_name}…")
        window_text = self.palette().color(QPalette.ColorRole.WindowText).name()
        self._status_label.setStyleSheet(f"color: {window_text};")
        self._status_label.show()
        self._progress_bar.show()
        self._recents_container.hide()
        for w in self._interactive_widgets:
            w.setEnabled(False)

    def set_error(self, message: str) -> None:
        self._status_label.setText(f"Couldn't open this project — {message}")
        self._status_label.setStyleSheet(f"color: {Colors.ERROR};")
        self._status_label.show()
        self._progress_bar.hide()
        if self._recent_projects:
            self._recents_container.show()
        for w in self._interactive_widgets:
            w.setEnabled(True)
        self._open_button.setFocus()
