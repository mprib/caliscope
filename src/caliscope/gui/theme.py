"""Centralized visual theme for Caliscope Qt widgets.

This module provides consistent styling across the application:
- Colors: Semantic color constants for the dark theme
- Typography: Common label styling patterns
- Styles: Pre-composed Qt stylesheets for widgets

Usage:
    from caliscope.gui.theme import Colors, Typography, Styles

    button.setStyleSheet(Styles.PRIMARY_BUTTON)
    label.setStyleSheet(Typography.SECTION_HEADER)
"""

from __future__ import annotations


class Colors:
    """Semantic color palette for dark theme."""

    # Primary interactive (blue accent)
    PRIMARY = "#0078d4"
    PRIMARY_HOVER = "#106ebe"
    PRIMARY_PRESSED = "#005a9e"

    # Status indicators (Material Design inspired)
    SUCCESS = "#4CAF50"  # Green - complete, calibrated
    WARNING = "#FFA000"  # Amber - in progress, partial
    ERROR = "#F44336"  # Red - failed, needs attention

    # Surfaces (dark theme, darkest to lightest)
    SURFACE_DARK = "#1a1a1a"  # Video backgrounds, deep panels
    SURFACE = "#2a2a2a"  # Default panel background
    SURFACE_LIGHT = "#3a3a3a"  # Raised elements, grooves

    # Text hierarchy
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_MUTED = "#888888"  # Headers, helper text
    TEXT_DISABLED = "#555555"

    # Borders
    BORDER_SUBTLE = "#333333"  # Grid lines, subtle dividers
    BORDER = "#555555"  # Default borders


class Typography:
    """Pre-composed label styles for common patterns.

    Only includes patterns appearing in 3+ places across the codebase.
    """

    SECTION_HEADER = f"font-weight: bold; color: {Colors.TEXT_MUTED}; font-size: 11px;"
    HELPER_TEXT = f"color: {Colors.TEXT_MUTED}; font-style: italic;"


class Styles:
    """Pre-composed Qt stylesheets for widgets.

    Each style includes all relevant pseudo-states:
    - :hover - mouse over
    - :pressed - mouse down / active
    - :disabled - widget disabled
    """

    PRIMARY_BUTTON = f"""
        QPushButton {{
            background-color: {Colors.PRIMARY};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 20px;
            font-weight: bold;
            min-width: 100px;
        }}
        QPushButton:hover {{
            background-color: {Colors.PRIMARY_HOVER};
        }}
        QPushButton:pressed {{
            background-color: {Colors.PRIMARY_PRESSED};
        }}
        QPushButton:disabled {{
            background-color: {Colors.TEXT_DISABLED};
            color: {Colors.TEXT_MUTED};
        }}
    """

    GHOST_BUTTON = f"""
        QPushButton {{
            background-color: transparent;
            color: {Colors.PRIMARY};
            border: 1px solid {Colors.PRIMARY};
            border-radius: 4px;
            padding: 4px 12px;
        }}
        QPushButton:hover {{
            background-color: rgba(0, 120, 212, 0.15);
        }}
        QPushButton:pressed {{
            background-color: rgba(0, 120, 212, 0.25);
        }}
        QPushButton:disabled {{
            color: {Colors.TEXT_DISABLED};
            border-color: {Colors.BORDER_SUBTLE};
            background-color: transparent;
        }}
    """

    SLIDER = f"""
        QSlider::groove:horizontal {{
            height: 8px;
            background: {Colors.SURFACE_LIGHT};
            border-radius: 4px;
        }}
        QSlider::groove:horizontal:disabled {{
            background: {Colors.BORDER_SUBTLE};
        }}
        QSlider::handle:horizontal {{
            width: 20px;
            height: 20px;
            margin: -6px 0;
            background: {Colors.PRIMARY};
            border-radius: 10px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {Colors.PRIMARY_HOVER};
        }}
        QSlider::handle:horizontal:pressed {{
            background: {Colors.PRIMARY_PRESSED};
        }}
        QSlider::handle:horizontal:disabled {{
            background: {Colors.TEXT_DISABLED};
        }}
    """
