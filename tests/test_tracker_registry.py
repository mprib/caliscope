"""Tests for tracker_registry module."""

from pathlib import Path

import pytest

from caliscope.trackers import tracker_registry


class TestOnnxScanning:
    """Tests for ONNX model scanning."""

    def test_scan_nonexistent_dir_safe(self):
        """Scanning nonexistent directory doesn't raise."""
        tracker_registry.scan_onnx_models(Path("/nonexistent"))

    def test_scan_without_onnxruntime_safe(self, monkeypatch):
        """Scanning without onnxruntime installed doesn't raise."""
        # This test assumes onnxruntime may or may not be installed
        # Just verify scan_onnx_models is callable and doesn't crash
        tracker_registry.scan_onnx_models(Path("/tmp"))


class TestRegistryCore:
    """Tests for core registry operations."""

    def test_is_registered_false_for_unknown(self):
        """is_registered() returns False for unknown keys."""
        assert not tracker_registry.is_registered("NONEXISTENT")

    def test_create_unknown_tracker_raises(self):
        """Creating unknown tracker raises KeyError."""
        with pytest.raises(KeyError, match="Unknown tracker"):
            tracker_registry.create("NONEXISTENT")

    def test_available_names_returns_list(self):
        """available_names() returns a list (possibly empty)."""
        result = tracker_registry.available_names()
        assert isinstance(result, list)

    def test_clear_removes_all(self):
        """clear() removes all registrations."""
        tracker_registry.clear()
        assert tracker_registry.available_names() == []

    def test_register_and_retrieve(self):
        """Can register a tracker and retrieve it."""
        from caliscope.core.charuco import Charuco
        from caliscope.trackers.charuco_tracker import CharucoTracker

        charuco_toml = Path(__file__).parent / "sessions" / "post_optimization" / "charuco.toml"
        charuco = Charuco.from_toml(charuco_toml)

        tracker_registry.clear()
        tracker_registry.register("CHARUCO_TEST", lambda: CharucoTracker(charuco), display_name="Charuco Test")

        assert tracker_registry.is_registered("CHARUCO_TEST")
        assert "CHARUCO_TEST" in tracker_registry.available_names()
        assert tracker_registry.display_name_for("CHARUCO_TEST") == "Charuco Test"

        tracker = tracker_registry.create("CHARUCO_TEST")
        assert tracker.name == "CHARUCO"

        tracker_registry.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
