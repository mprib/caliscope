"""Tracker registry — maps string keys to zero-arg factory callables.

Built-in trackers register at import time. ONNX trackers register via
scan_onnx_models() called at app startup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from caliscope.tracker import Tracker, WireFrameView

if TYPE_CHECKING:
    from caliscope.trackers.model_card import ModelCard

logger = logging.getLogger(__name__)

# Module-level state — the registry
_factories: dict[str, Callable[[], Tracker]] = {}
_display_names: dict[str, str] = {}
_wireframes: dict[str, WireFrameView | None] = {}
_model_cards: dict[str, ModelCard] = {}


def register(
    key: str,
    factory: Callable[[], Tracker],
    display_name: str | None = None,
    wireframe: WireFrameView | None = None,
    model_card: ModelCard | None = None,
) -> None:
    """Register a tracker factory under a filesystem-safe key.

    Overwrites silently if key already exists (allows re-scanning ONNX dir).
    """
    _factories[key] = factory
    _display_names[key] = display_name if display_name is not None else key.replace("_", " ").title()
    _wireframes[key] = wireframe
    if model_card is not None:
        _model_cards[key] = model_card


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


def is_model_ready(key: str) -> bool:
    """Check if a registered tracker's model weights are available.

    Returns True for built-in trackers (always ready).
    For ONNX trackers, delegates to ModelCard.onnx_exists (live filesystem check).
    """
    card = _model_cards.get(key)
    if card is None:
        return True  # Built-in tracker, always ready
    return card.onnx_exists


def model_card_for(key: str) -> ModelCard | None:
    """Return the ModelCard for an ONNX tracker, or None for built-ins."""
    return _model_cards.get(key)


def scan_onnx_models(models_dir: Path) -> None:
    """Scan directory for .toml model cards and register each as ONNX tracker.

    Logs warnings for malformed .toml files but does not raise.
    """
    if not models_dir.exists():
        logger.debug("Models directory not found: %s", models_dir)
        return

    from caliscope.trackers.model_card import ModelCard
    from caliscope.trackers.onnx_tracker import OnnxTracker

    for toml_path in sorted(models_dir.glob("*.toml")):
        try:
            card = ModelCard.from_toml(toml_path, models_dir=models_dir)
        except Exception as e:
            logger.warning("Skipping malformed model card %s: %s", toml_path, e)
            continue
        key = f"ONNX_{card.model_path.stem}"
        register(
            key,
            lambda card=card: OnnxTracker(card),
            display_name=card.name,
            wireframe=card.wireframe,
            model_card=card,
        )
        logger.info("Registered ONNX tracker: %s (%s)", key, card.name)


def clear() -> None:
    """Remove all registrations. For testing only."""
    _factories.clear()
    _display_names.clear()
    _wireframes.clear()
    _model_cards.clear()


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
