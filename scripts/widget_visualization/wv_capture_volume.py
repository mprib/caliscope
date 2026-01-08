"""Visual test for CaptureVolumeVisualizer (3D OpenGL rendering).

Widget visualization script testing the 3D capture volume display.
Loads a calibrated project, creates the visualizer, and captures
screenshots of the OpenGL scene with camera frustums and point cloud.

Usage:
    python scripts/widget_visualization/wv_capture_volume.py

Or headless:
    xvfb-run python scripts/widget_visualization/wv_capture_volume.py
"""

import sys
from pathlib import Path

# Add project src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from caliscope.core.capture_volume.capture_volume import CaptureVolume  # noqa: E402
from caliscope.gui.vizualize.calibration.capture_volume_visualizer import (  # noqa: E402
    CaptureVolumeVisualizer,
)
from caliscope.repositories import (  # noqa: E402
    CameraArrayRepository,
    CaptureVolumeRepository,
)

from utils import capture_widget, clear_output_dir, process_events_for  # noqa: E402

# Use sample project which is fully calibrated
SAMPLE_PROJECT = Path("/home/mprib/caliscope_projects/new_minimal_project")


def main():
    clear_output_dir()

    if not SAMPLE_PROJECT.exists():
        print(f"ERROR: Sample project not found at {SAMPLE_PROJECT}")
        sys.exit(1)

    print(f"Using project: {SAMPLE_PROJECT}")

    # Initialize Qt application
    QApplication(sys.argv)

    # Load camera array
    camera_repository = CameraArrayRepository(SAMPLE_PROJECT / "camera_array.toml")
    camera_array = camera_repository.load()
    print(f"Loaded camera array with {len(camera_array.cameras)} cameras")

    # Load capture volume data
    cv_repository = CaptureVolumeRepository(SAMPLE_PROJECT)
    if not cv_repository.exists():
        print("ERROR: Capture volume data not found in project")
        sys.exit(1)

    point_estimates = cv_repository.load_point_estimates()
    metadata = cv_repository.load_metadata()
    print(f"Loaded point estimates with {len(point_estimates.obj)} points")

    # Create CaptureVolume
    capture_volume = CaptureVolume(
        camera_array=camera_array,
        _point_estimates=point_estimates,
        stage=metadata.get("stage", 0),
        origin_sync_index=metadata.get("origin_sync_index"),
    )

    # Create the visualizer
    visualizer = CaptureVolumeVisualizer(capture_volume=capture_volume)
    visualizer.scene.resize(800, 600)
    visualizer.scene.show()

    # OpenGL needs extra time to initialize
    process_events_for(1500)

    # Step 1: Initial view
    capture_widget(visualizer.scene, "01_initial_3d.png")
    print("Step 1: Initial 3D view captured")

    # Step 2: Display points at first sync index
    if visualizer.point_estimates is not None:
        first_sync = visualizer.min_sync_index
        visualizer.display_points(first_sync)
        process_events_for(500)
        capture_widget(visualizer.scene, "02_with_points.png")
        print(f"Step 2: Points displayed at sync_index={first_sync}")

        # Step 3: Move to middle sync index
        mid_sync = (visualizer.min_sync_index + visualizer.max_sync_index) // 2
        visualizer.display_points(mid_sync)
        process_events_for(500)
        capture_widget(visualizer.scene, "03_mid_sync.png")
        print(f"Step 3: Points at mid sync_index={mid_sync}")

        # Step 4: Move to last sync index
        last_sync = visualizer.max_sync_index
        visualizer.display_points(last_sync)
        process_events_for(500)
        capture_widget(visualizer.scene, "04_last_sync.png")
        print(f"Step 4: Points at last sync_index={last_sync}")

    # Step 5: Rotate camera view
    visualizer.scene.setCameraPosition(azimuth=45, elevation=30, distance=5)
    process_events_for(500)
    capture_widget(visualizer.scene, "05_rotated_view.png")
    print("Step 5: Rotated camera view (azimuth=45, elevation=30)")

    # Step 6: Different rotation
    visualizer.scene.setCameraPosition(azimuth=135, elevation=60, distance=3)
    process_events_for(500)
    capture_widget(visualizer.scene, "06_top_view.png")
    print("Step 6: Top-ish view (azimuth=135, elevation=60)")

    print("\n" + "=" * 50)
    print("CAPTURE VOLUME TEST COMPLETE")
    print("=" * 50)
    print("\nScreenshots saved to: scripts/widget_visualization/output/")
    print("\nVerification checklist:")
    print("  [ ] 01: 3D scene renders with axis and camera frustums")
    print("  [ ] 02: White point cloud visible in scene")
    print("  [ ] 03: Points moved (different position in capture volume)")
    print("  [ ] 04: Points at different position again")
    print("  [ ] 05: View rotated (different angle)")
    print("  [ ] 06: View from above (higher elevation)")
    print()


if __name__ == "__main__":
    main()
