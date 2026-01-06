"""Full app workflow smoke test for Caliscope.

Ralph Wiggum visual testing script that:
1. Launches the full application with a sample project
2. Navigates to the Camera tab
3. Plays video, captures screenshot
4. Pauses, moves slider, captures screenshot
5. Clicks autocalibrate, waits 4 seconds, captures screenshot
6. Navigates to Capture Volume tab (if enabled), captures screenshot

Uses QTimer.singleShot() for sequencing async operations.

Usage:
    xvfb-run python scripts/ralph_wiggum_viz/rw_full_workflow.py

Or with a display:
    python scripts/ralph_wiggum_viz/rw_full_workflow.py
"""

import sys
from pathlib import Path

# Add project src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from caliscope.cameras.camera_array import CameraArray  # noqa: E402
from caliscope.gui.main_widget import MainWindow  # noqa: E402
from caliscope.gui.vizualize.calibration.capture_volume_visualizer import (  # noqa: E402
    CaptureVolumeVisualizer,
)

from utils import capture_widget, clear_output_dir  # noqa: E402

# Sample project path - fully calibrated
SAMPLE_PROJECT = Path("/home/mprib/caliscope_projects/new_minimal_project")


class FullWorkflowTest:
    """Orchestrates the full workflow test sequence."""

    def __init__(self, window: MainWindow, project_path: Path):
        self.window = window
        self.project_path = project_path
        self.step = 0

    def run(self):
        """Start the test sequence."""
        print(f"Loading project: {self.project_path}")

        # Launch workspace first (this creates the controller)
        self.window.launch_workspace(str(self.project_path))

        # Now connect to workspace load completion
        # (controller exists after launch_workspace is called)
        self.window.controller.load_workspace_thread.finished.connect(self.on_workspace_loaded)

    def on_workspace_loaded(self):
        """Called when workspace finishes loading asynchronously."""
        print("Workspace loaded, starting test sequence...")

        # Small delay to let UI fully render
        QTimer.singleShot(500, self.capture_initial)

    def capture_initial(self):
        """Step 1: Capture initial main window state."""
        capture_widget(self.window, "01_main_initial.png")
        print("Step 1: Initial state captured")

        # Navigate to Cameras tab
        cameras_index = self.window.find_tab_index_by_title("Cameras")
        if cameras_index >= 0 and self.window.central_tab.isTabEnabled(cameras_index):
            self.window.central_tab.setCurrentIndex(cameras_index)
            QTimer.singleShot(500, self.capture_cameras_tab)
        else:
            print("WARNING: Cameras tab not found or not enabled")
            self.try_capture_volume()

    def capture_cameras_tab(self):
        """Step 2: Capture Cameras tab."""
        capture_widget(self.window, "02_cameras_tab.png")
        print("Step 2: Cameras tab captured")

        # Get the first camera's playback widget
        tab_widget = self.window.intrinsic_cal_widget.tabWidget
        if tab_widget.count() > 0:
            first_cam_widget = tab_widget.widget(0)

            # Click play button
            print("Clicking play button...")
            first_cam_widget.play_button.click()
            QTimer.singleShot(1000, self.capture_playing)
        else:
            print("WARNING: No camera widgets found")
            self.try_capture_volume()

    def capture_playing(self):
        """Step 3: Capture video playing state."""
        capture_widget(self.window, "03_video_playing.png")
        print("Step 3: Video playing captured")

        # Pause and move slider
        tab_widget = self.window.intrinsic_cal_widget.tabWidget
        first_cam_widget = tab_widget.widget(0)

        # Pause
        print("Pausing video...")
        first_cam_widget.play_button.click()

        # Move slider to ~1/3 position
        slider = first_cam_widget.slider
        target_pos = slider.minimum() + (slider.maximum() - slider.minimum()) // 3
        slider.setValue(target_pos)

        QTimer.singleShot(500, self.capture_slider_moved)

    def capture_slider_moved(self):
        """Step 4: Capture after slider moved."""
        capture_widget(self.window, "04_slider_moved.png")
        print("Step 4: Slider moved captured")

        # Click autocalibrate button
        tab_widget = self.window.intrinsic_cal_widget.tabWidget
        first_cam_widget = tab_widget.widget(0)

        print("Clicking autocalibrate button...")
        first_cam_widget.autocalibrate_btn.click()

        # Wait 4 seconds for autocalibration to progress
        QTimer.singleShot(4000, self.capture_autocalibrating)

    def capture_autocalibrating(self):
        """Step 5: Capture during autocalibration."""
        capture_widget(self.window, "05_autocalibrating.png")
        print("Step 5: Autocalibrating captured (4 seconds after click)")

        # Try to navigate to Capture Volume tab
        self.try_capture_volume()

    def try_capture_volume(self):
        """Step 6: Try to capture Capture Volume tab if enabled."""
        cv_index = self.window.find_tab_index_by_title("Capture Volume")
        if cv_index >= 0 and self.window.central_tab.isTabEnabled(cv_index):
            self.window.central_tab.setCurrentIndex(cv_index)
            # Longer delay for OpenGL initialization
            QTimer.singleShot(1500, self.capture_capture_volume)
        else:
            print("Capture Volume tab not enabled (expected if not calibrated)")
            self.finish()

    def capture_capture_volume(self):
        """Step 6: Capture Capture Volume tab."""
        capture_widget(self.window, "06_capture_volume_tab.png")
        print("Step 6: Capture Volume tab captured")
        self.finish()

    def finish(self):
        """Test complete."""
        print("\n" + "=" * 50)
        print("RALPH WIGGUM WORKFLOW TEST COMPLETE")
        print("=" * 50)
        print("\nScreenshots saved to: scripts/ralph_wiggum/gui/output/")
        print("\nVerification checklist:")
        print("  [ ] 01: Main window renders, tabs visible")
        print("  [ ] 02: Cameras tab shows camera sub-tabs, video frame visible")
        print("  [ ] 03: Video frame has changed (different from 02)")
        print("  [ ] 04: Frame changed after slider move (different position)")
        print("  [ ] 05: Frame changed during autocalibration (grid overlay may appear)")
        print("  [ ] 06: 3D capture volume renders (if tab was enabled)")
        print()

        QApplication.instance().quit()


def main():
    clear_output_dir()

    if not SAMPLE_PROJECT.exists():
        print(f"ERROR: Sample project not found at {SAMPLE_PROJECT}")
        print("Please ensure the project exists or update SAMPLE_PROJECT path.")
        sys.exit(1)

    print(f"Project directory: {SAMPLE_PROJECT}")

    # Initialize Qt application
    app = QApplication(sys.argv)

    # Warm up OpenGL (same as launch_main does)
    dummy_widget = CaptureVolumeVisualizer(camera_array=CameraArray({}))
    del dummy_widget

    # Create main window
    window = MainWindow()
    window.resize(1200, 800)
    window.show()

    # Create and run test
    test = FullWorkflowTest(window, SAMPLE_PROJECT)

    # Start test after window is shown
    QTimer.singleShot(100, test.run)

    # Run event loop
    app.exec()


if __name__ == "__main__":
    main()
