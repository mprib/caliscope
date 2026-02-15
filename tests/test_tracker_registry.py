"""Tests for tracker_registry module."""

from pathlib import Path

import pytest

from caliscope.trackers import tracker_registry


class TestBuiltinTrackers:
    """Tests for built-in tracker registration."""

    def test_builtins_registered_at_import(self):
        """Built-in trackers should be registered on module import."""
        available = tracker_registry.available_names()
        assert "HAND" in available
        assert "POSE" in available
        assert "SIMPLE_HOLISTIC" in available
        assert "HOLISTIC" in available

    def test_create_hand_tracker(self):
        """Can create Hand tracker instance."""
        tracker = tracker_registry.create("HAND")
        assert tracker.name == "HAND"

    def test_create_pose_tracker(self):
        """Can create Pose tracker instance."""
        tracker = tracker_registry.create("POSE")
        assert tracker.name == "POSE"

    def test_display_names(self):
        """Display names are human-readable."""
        assert tracker_registry.display_name_for("HAND") == "Hand"
        assert tracker_registry.display_name_for("POSE") == "Pose"
        assert tracker_registry.display_name_for("SIMPLE_HOLISTIC") == "Simple Holistic"
        assert tracker_registry.display_name_for("HOLISTIC") == "Holistic"

    def test_is_registered(self):
        """is_registered() checks for key presence."""
        assert tracker_registry.is_registered("HAND")
        assert not tracker_registry.is_registered("NONEXISTENT")

    def test_create_unknown_tracker_raises(self):
        """Creating unknown tracker raises KeyError."""
        with pytest.raises(KeyError, match="Unknown tracker"):
            tracker_registry.create("NONEXISTENT")


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


class TestRegistryClear:
    """Tests for clear() testing utility."""

    def test_clear_removes_all(self):
        """clear() removes all registrations."""
        tracker_registry.clear()
        assert tracker_registry.available_names() == []

        # Re-register builtins for other tests
        tracker_registry._register_builtins()

    def test_clear_and_reregister(self):
        """Can clear and re-register."""
        initial = set(tracker_registry.available_names())
        tracker_registry.clear()
        tracker_registry._register_builtins()
        after = set(tracker_registry.available_names())
        assert initial == after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
