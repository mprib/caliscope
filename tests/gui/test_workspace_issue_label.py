"""Canary for the shared workspace issue label."""

from caliscope.core.workflow_status import WorkspaceIssue
from caliscope.gui.widgets.workspace_issue_label import WorkspaceIssueLabel


def test_label_shows_issues_and_hides_when_there_is_nothing_to_say(qapp) -> None:
    label = WorkspaceIssueLabel()
    assert label.isHidden()

    label.set_issues([WorkspaceIssue(code="x", message="Missing a.mp4.", relative_path="a.mp4")])
    assert label.text() == "Missing a.mp4."

    label.set_issues([])
    assert label.isHidden()

    label.set_issues([], empty_text="Nothing here")
    assert label.text() == "Nothing here"
