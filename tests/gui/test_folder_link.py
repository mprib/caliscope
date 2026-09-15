"""Tests for contextual local-folder navigation links."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from caliscope.gui.widgets.folder_link import FolderLink


def test_folder_link_opens_current_existing_folder(tmp_path: Path, qapp, monkeypatch) -> None:
    link = FolderLink("Open source folder", tmp_path)
    opened = []
    monkeypatch.setattr(
        "caliscope.gui.widgets.folder_link.QDesktopServices.openUrl",
        lambda url: opened.append(url),
    )
    link.show()

    QTest.mouseClick(link, Qt.MouseButton.LeftButton)

    assert link.folder == tmp_path
    assert link.isEnabled()
    assert Path(opened[0].toLocalFile()) == tmp_path


def test_folder_link_keyboard_activation_and_missing_target(tmp_path: Path, qapp, monkeypatch) -> None:
    link = FolderLink("Open source folder", tmp_path)
    opened = []
    monkeypatch.setattr(
        "caliscope.gui.widgets.folder_link.QDesktopServices.openUrl",
        lambda url: opened.append(url),
    )
    link.show()
    link.setFocus()

    QTest.keyClick(link, Qt.Key.Key_Return)
    QTest.keyClick(link, Qt.Key.Key_Space)

    assert [Path(url.toLocalFile()) for url in opened] == [tmp_path, tmp_path]

    non_folder = tmp_path / "not-a-folder"
    non_folder.write_text("not a directory")
    link.set_folder(non_folder)
    QTest.mouseClick(link, Qt.MouseButton.LeftButton)

    assert link.folder == non_folder
    assert not link.isEnabled()
    assert link.toolTip() == f"Folder unavailable: {non_folder}"
    assert len(opened) == 2


def test_folder_link_does_not_open_target_deleted_after_setup(tmp_path: Path, qapp, monkeypatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = FolderLink("Open source folder", target)
    opened = []
    monkeypatch.setattr(
        "caliscope.gui.widgets.folder_link.QDesktopServices.openUrl",
        lambda url: opened.append(url),
    )
    target.rmdir()

    link.show()
    QTest.mouseClick(link, Qt.MouseButton.LeftButton)

    assert opened == []
