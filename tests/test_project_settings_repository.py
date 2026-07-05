from pathlib import Path

from caliscope.repositories.project_settings_repository import ProjectSettingsRepository


def test_refine_intrinsics_defaults_true(tmp_path: Path) -> None:
    repo = ProjectSettingsRepository(tmp_path / "project_settings.toml")
    assert repo.get_refine_intrinsics() is True


def test_refine_intrinsics_persists(tmp_path: Path) -> None:
    settings_path = tmp_path / "project_settings.toml"
    repo = ProjectSettingsRepository(settings_path)

    repo.set_refine_intrinsics(False)
    assert repo.get_refine_intrinsics() is False

    reloaded = ProjectSettingsRepository(settings_path)
    assert reloaded.get_refine_intrinsics() is False


def test_origin_object_id_defaults_none(tmp_path: Path) -> None:
    repo = ProjectSettingsRepository(tmp_path / "project_settings.toml")
    assert repo.get_origin_object_id() is None


def test_origin_object_id_set_and_read_back(tmp_path: Path) -> None:
    settings_path = tmp_path / "project_settings.toml"
    repo = ProjectSettingsRepository(settings_path)

    repo.set_origin_object_id(4)
    assert repo.get_origin_object_id() == 4

    reloaded = ProjectSettingsRepository(settings_path)
    assert reloaded.get_origin_object_id() == 4


def test_origin_object_id_none_removes_key_from_toml(tmp_path: Path) -> None:
    settings_path = tmp_path / "project_settings.toml"
    repo = ProjectSettingsRepository(settings_path)

    repo.set_origin_object_id(4)
    repo.set_origin_object_id(None)

    assert repo.get_origin_object_id() is None
    assert "origin_object_id" not in settings_path.read_text()


def test_origin_sync_index_defaults_none(tmp_path: Path) -> None:
    repo = ProjectSettingsRepository(tmp_path / "project_settings.toml")
    assert repo.get_origin_sync_index() is None


def test_origin_sync_index_set_and_read_back(tmp_path: Path) -> None:
    settings_path = tmp_path / "project_settings.toml"
    repo = ProjectSettingsRepository(settings_path)

    repo.set_origin_sync_index(87)
    assert repo.get_origin_sync_index() == 87

    reloaded = ProjectSettingsRepository(settings_path)
    assert reloaded.get_origin_sync_index() == 87


def test_origin_sync_index_none_removes_key_from_toml(tmp_path: Path) -> None:
    settings_path = tmp_path / "project_settings.toml"
    repo = ProjectSettingsRepository(settings_path)

    repo.set_origin_sync_index(87)
    repo.set_origin_sync_index(None)

    assert repo.get_origin_sync_index() is None
    assert "origin_sync_index" not in settings_path.read_text()


def test_round_trip_all_three_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / "project_settings.toml"
    repo = ProjectSettingsRepository(settings_path)

    repo.set_refine_intrinsics(False)
    repo.set_origin_object_id(4)
    repo.set_origin_sync_index(87)

    reloaded = ProjectSettingsRepository(settings_path)
    assert reloaded.get_refine_intrinsics() is False
    assert reloaded.get_origin_object_id() == 4
    assert reloaded.get_origin_sync_index() == 87
