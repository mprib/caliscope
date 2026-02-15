"""Tracker registry — maps string keys to zero-arg factory callables.

Built-in trackers register at import time. ONNX trackers register via
scan_onnx_models() called at app startup.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from caliscope.tracker import Tracker, WireFrameView

logger = logging.getLogger(__name__)

# Module-level state — the registry
_factories: dict[str, Callable[[], Tracker]] = {}
_display_names: dict[str, str] = {}
_wireframes: dict[str, WireFrameView | None] = {}


def register(
    key: str, factory: Callable[[], Tracker], display_name: str | None = None, wireframe: WireFrameView | None = None
) -> None:
    """Register a tracker factory under a filesystem-safe key.

    Overwrites silently if key already exists (allows re-scanning ONNX dir).
    """
    _factories[key] = factory
    _display_names[key] = display_name if display_name is not None else key.replace("_", " ").title()
    _wireframes[key] = wireframe


def create(key: str) -> Tracker:
    """Construct a fresh Tracker instance by registry key.

    Raises KeyError if key not registered.
    """
    try:
        return _factories[key]()
    except KeyError:
        raise KeyError(f"Unknown tracker: {key!r}. Registered: {list(_factories)}") from None


def available_names() -> list[str]:
    """Return sorted list of registered tracker keys."""
    return sorted(_factories.keys())


def display_name_for(key: str) -> str:
    """Human-readable label for GUI display."""
    return _display_names.get(key, key.replace("_", " ").title())


def is_registered(key: str) -> bool:
    """Check if a tracker name is registered."""
    return key in _factories


def wireframe_for(key: str) -> WireFrameView | None:
    """Return wireframe topology for a registered tracker, or None."""
    return _wireframes.get(key)


def scan_onnx_models(models_dir: Path) -> None:
    """Scan directory for .toml model cards and register each as ONNX tracker.

    Skips entirely if onnxruntime is not installed.
    Logs warnings for malformed .toml files but does not raise.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.debug("onnxruntime not installed, skipping ONNX model scan")
        return

    if not models_dir.exists():
        logger.debug("Models directory not found: %s", models_dir)
        return

    from caliscope.trackers.model_card import ModelCard
    from caliscope.trackers.onnx_tracker import OnnxTracker

    for toml_path in sorted(models_dir.glob("*.toml")):
        try:
            card = ModelCard.from_toml(toml_path)
        except Exception as e:
            logger.warning("Skipping malformed model card %s: %s", toml_path, e)
            continue
        key = f"ONNX_{card.model_path.stem}"
        register(key, lambda card=card: OnnxTracker(card), display_name=card.name, wireframe=card.wireframe)
        logger.info("Registered ONNX tracker: %s (%s)", key, card.name)


def clear() -> None:
    """Remove all registrations. For testing only."""
    _factories.clear()
    _display_names.clear()
    _wireframes.clear()


def _register_builtins() -> None:
    """Register the 4 built-in reconstruction trackers."""
    from pathlib import Path

    from caliscope.trackers.hand_tracker import HandTracker
    from caliscope.trackers.holistic.holistic_tracker import POINT_NAMES as HOLISTIC_POINT_NAMES
    from caliscope.trackers.holistic.holistic_tracker import HolisticTracker
    from caliscope.trackers.pose_tracker import PoseTracker
    from caliscope.trackers.simple_holistic_tracker import SimpleHolisticTracker
    from caliscope.trackers.wireframe_builder import get_wireframe

    # Load holistic wireframe for registration metadata
    holistic_wireframe_path = Path(Path(__file__).parent.parent, "gui/geometry/wireframes/holistic_wireframe.toml")
    # Invert POINT_NAMES from {id: name} to {name: id} for WireFrameView
    point_names_inverted = {name: pid for pid, name in HOLISTIC_POINT_NAMES.items()}
    holistic_wireframe = get_wireframe(holistic_wireframe_path, point_names_inverted)

    register("HAND", HandTracker, display_name="Hand")
    register("POSE", PoseTracker, display_name="Pose")
    register("SIMPLE_HOLISTIC", SimpleHolisticTracker, display_name="Simple Holistic")
    register("HOLISTIC", HolisticTracker, display_name="Holistic", wireframe=holistic_wireframe)


# Auto-register builtins on import
_register_builtins()
