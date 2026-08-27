"""Shared label for rendering workspace file issues."""

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from caliscope.core.workflow_status import WorkspaceIssue
from caliscope.gui.theme import Colors


class WorkspaceIssueLabel(QLabel):
    """Warning text for workspace file problems, hidden when there is nothing to say."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceIssueLabel")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setStyleSheet(f"color: {Colors.WARNING};")
        self.hide()

    def set_issues(self, issues: Iterable[WorkspaceIssue], empty_text: str = "") -> None:
        """Render one line per issue, falling back to empty_text when there are none.

        Hides itself when the resulting text is empty.
        """
        text = "\n".join(issue.message for issue in issues)
        if not text:
            text = empty_text

        if text:
            self.setText(text)
            self.show()
        else:
            self.clear()
            self.hide()
