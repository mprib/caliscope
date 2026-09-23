"""Unit tests for WelcomeWidget."""

from caliscope.gui.widgets.welcome_widget import WelcomeWidget


def test_construction_empty_recents(qapp):
    w = WelcomeWidget(recent_projects=[])
    assert w._recents_container.isHidden()
    assert w._open_button.isEnabled()


def test_construction_populated_recents(qapp, tmp_path):
    paths = []
    for i in range(3):
        (tmp_path / f"project_{i}").mkdir()
        paths.append(str(tmp_path / f"project_{i}"))

    w = WelcomeWidget(recent_projects=paths)
    assert not w._recents_container.isHidden()
    assert len(w._recent_links) == 3


def test_open_project_signal(qapp):
    w = WelcomeWidget(recent_projects=[])
    received = []
    w.open_project_requested.connect(lambda: received.append(True))
    w._open_button.click()
    assert received == [True]


def test_recent_project_signal(qapp, tmp_path):
    project = tmp_path / "my_project"
    project.mkdir()
    path_str = str(project)

    w = WelcomeWidget(recent_projects=[path_str])
    received = []
    w.recent_project_selected.connect(lambda p: received.append(p))
    w._recent_links[0].clicked.emit()
    assert received == [path_str]


def test_set_loading_disables_controls(qapp, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()

    w = WelcomeWidget(recent_projects=[str(project)])
    w.set_loading(str(project))

    assert not w._open_button.isEnabled()
    assert not w._progress_bar.isHidden()
    assert not w._status_label.isHidden()
    assert "Loading" in w._status_label.text()
    for link in w._recent_links:
        assert not link.isEnabled()


def test_set_error_reenables_controls(qapp, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()

    w = WelcomeWidget(recent_projects=[str(project)])
    w.set_loading(str(project))
    w.set_error("bad things happened")

    assert w._open_button.isEnabled()
    assert w._progress_bar.isHidden()
    assert not w._status_label.isHidden()
    assert "bad things happened" in w._status_label.text()
    for link in w._recent_links:
        assert link.isEnabled()
