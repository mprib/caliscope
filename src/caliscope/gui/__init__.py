from pathlib import Path

from caliscope.gui.capture_volume_viewer import view_capture_volume

ICONS_DIR = Path(__file__).parent / "icons"


__all__ = ["ICONS_DIR", "view_capture_volume"]
