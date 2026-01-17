"""Visual test for CharucoWidget interactions.

Widget visualization script demonstrating widget interaction testing pattern.
Creates the CharucoWidget and interacts with spinboxes/checkboxes to
change parameters, capturing screenshots at each state.

Usage:
    python scripts/widget_visualization/wv_charuco_widget.py

Or headless:
    xvfb-run python scripts/widget_visualization/wv_charuco_widget.py
"""

import sys
from pathlib import Path

# Add project src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from caliscope.workspace_coordinator import WorkspaceCoordinator  # noqa: E402
from caliscope.gui.charuco_widget import CharucoWidget  # noqa: E402

from utils import capture_widget, clear_output_dir, process_events_for  # noqa: E402

# Use sample project which has config files
SAMPLE_PROJECT = Path("/home/mprib/caliscope_projects/new_minimal_project")


def main():
    clear_output_dir()

    if not SAMPLE_PROJECT.exists():
        print(f"ERROR: Sample project not found at {SAMPLE_PROJECT}")
        sys.exit(1)

    print(f"Using project: {SAMPLE_PROJECT}")

    # Initialize Qt application
    QApplication(sys.argv)

    # Create coordinator (loads charuco config from project)
    coordinator = WorkspaceCoordinator(SAMPLE_PROJECT)

    # Create the CharucoWidget
    widget = CharucoWidget(coordinator)
    widget.resize(600, 700)
    widget.show()

    process_events_for(500)  # Let it render

    # Step 1: Capture initial state
    capture_widget(widget, "01_initial.png")
    print(
        f"Step 1: Initial state - rows={widget.charuco_config.row_spin.value()}, "
        f"cols={widget.charuco_config.column_spin.value()}"
    )

    # Step 2: Change rows (interact with spinbox)
    original_rows = widget.charuco_config.row_spin.value()
    new_rows = original_rows + 2
    print(f"Step 2: Changing rows from {original_rows} to {new_rows}...")
    widget.charuco_config.row_spin.setValue(new_rows)
    process_events_for(300)
    capture_widget(widget, "02_more_rows.png")
    print(f"Step 2: Rows changed to {new_rows}")

    # Step 3: Change columns
    original_cols = widget.charuco_config.column_spin.value()
    new_cols = original_cols + 3
    print(f"Step 3: Changing columns from {original_cols} to {new_cols}...")
    widget.charuco_config.column_spin.setValue(new_cols)
    process_events_for(300)
    capture_widget(widget, "03_more_columns.png")
    print(f"Step 3: Columns changed to {new_cols}")

    # Step 4: Toggle invert checkbox
    was_inverted = widget.charuco_config.invert_checkbox.isChecked()
    print(f"Step 4: Toggling invert from {was_inverted} to {not was_inverted}...")
    widget.charuco_config.invert_checkbox.click()
    process_events_for(300)
    capture_widget(widget, "04_inverted.png")
    print(f"Step 4: Invert toggled to {widget.charuco_config.invert_checkbox.isChecked()}")

    # Step 5: Change board dimensions (width)
    original_width = widget.charuco_config.width_spin.value()
    new_width = original_width * 1.5
    print(f"Step 5: Changing width from {original_width} to {new_width}...")
    widget.charuco_config.width_spin.setValue(new_width)
    process_events_for(300)
    capture_widget(widget, "05_wider_board.png")
    print(f"Step 5: Width changed to {new_width}")

    # Step 6: Reset to something reasonable and toggle invert back
    print("Step 6: Resetting to clean state...")
    widget.charuco_config.row_spin.setValue(4)
    widget.charuco_config.column_spin.setValue(5)
    widget.charuco_config.width_spin.setValue(8.5)
    if widget.charuco_config.invert_checkbox.isChecked():
        widget.charuco_config.invert_checkbox.click()
    process_events_for(300)
    capture_widget(widget, "06_reset.png")
    print("Step 6: Reset to 4x5 standard board")

    print("\n" + "=" * 50)
    print("CHARUCO WIDGET TEST COMPLETE")
    print("=" * 50)
    print("\nScreenshots saved to: scripts/widget_visualization/output/")
    print("\nVerification checklist:")
    print("  [ ] 01: Initial charuco board renders with config controls visible")
    print("  [ ] 02: Board has more rows (taller grid)")
    print("  [ ] 03: Board has more columns (wider grid)")
    print("  [ ] 04: Board colors are inverted (white corners)")
    print("  [ ] 05: Board aspect ratio changed (wider)")
    print("  [ ] 06: Board reset to standard 4x5 non-inverted")
    print()


if __name__ == "__main__":
    main()
