"""Project setup and workflow status view.

This is the landing tab for Caliscope that:
1. Displays workspace path with folder access
2. Configures camera count and charuco board
3. Shows charuco board preview with PNG export
4. Displays workflow status checklist with navigation

Unlike other tabs, this view wires directly to the Coordinator (no Presenter)
because it only observes status - it doesn't orchestrate workflows.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from caliscope.core.workflow_status import StepStatus, WorkflowStatus
from caliscope.gui.widgets.charuco_config_panel import CharucoConfigPanel
from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)


class WorkflowStepRow(QWidget):
    """A single row in the workflow status checklist.

    Shows: [checkbox icon] Step Name         [status text] [Go to Tab]
    """

    navigation_requested = Signal(str)  # Tab name to navigate to

    def __init__(
        self,
        step_name: str,
        target_tab: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._step_name = step_name
        self._target_tab = target_tab

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Checkbox-style status indicator (read-only visual)
        self._status_icon = QLabel()
        self._status_icon.setFixedWidth(24)
        self._status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_icon)

        # Step name label
        self._name_label = QLabel(self._step_name)
        self._name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._name_label)

        layout.addStretch()

        # Status detail text
        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: #888;")
        self._status_label.setMinimumWidth(200)
        layout.addWidget(self._status_label)

        # Navigation button
        self._nav_btn = QPushButton("Go to Tab")
        self._nav_btn.setFixedWidth(90)
        layout.addWidget(self._nav_btn)

    def _connect_signals(self) -> None:
        self._nav_btn.clicked.connect(lambda: self.navigation_requested.emit(self._target_tab))

    def set_status(self, status: StepStatus, detail_text: str) -> None:
        """Update the row's visual state.

        Args:
            status: The step's completion status
            detail_text: Descriptive text (e.g., "4/4 cameras calibrated")
        """
        self._status_label.setText(detail_text)

        # Update checkbox-style icon based on status
        if status == StepStatus.COMPLETE:
            # Green checkmark
            self._status_icon.setText("[X]")
            self._status_icon.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self._name_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif status == StepStatus.INCOMPLETE:
            # Yellow partial/in-progress
            self._status_icon.setText("[~]")
            self._status_icon.setStyleSheet("color: #FFA000; font-weight: bold;")
            self._name_label.setStyleSheet("font-weight: bold; color: #FFA000;")
        elif status == StepStatus.AVAILABLE:
            # Blue indicator for optional capability
            self._status_icon.setText("[*]")
            self._status_icon.setStyleSheet("color: #2196F3; font-weight: bold;")
            self._name_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        else:  # NOT_STARTED
            # Gray empty
            self._status_icon.setText("[ ]")
            self._status_icon.setStyleSheet("color: #888; font-weight: bold;")
            self._name_label.setStyleSheet("font-weight: bold; color: #888;")


class ProjectSetupView(QWidget):
    """Project setup and status view (no Presenter needed).

    This view wires directly to the Coordinator because it only observes
    workflow status - it doesn't orchestrate any workflows. The status
    is recomputed on each refresh from ground truth in the Coordinator.

    Signal subscriptions:
    - status_changed: Workflow state may have changed (single refresh trigger)
    - charuco_changed: Board config updated (requires preview refresh)
    """

    tab_navigation_requested = Signal(str)  # Tab name to navigate to

    def __init__(self, coordinator: WorkspaceCoordinator) -> None:
        super().__init__()
        self._coordinator = coordinator

        self._setup_ui()
        self._connect_signals()
        self._refresh_status()
        self._update_charuco_preview()

        logger.info("ProjectSetupView created")

    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the widget layout."""
        # Use a scroll area for the main content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Row 1: Workspace path and folder button
        main_layout.addWidget(self._create_workspace_row())

        # Charuco configuration group
        main_layout.addWidget(self._create_charuco_group())

        # Workflow status group
        main_layout.addWidget(self._create_workflow_group())

        main_layout.addStretch()

        scroll.setWidget(content)

        # Set scroll area as main layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _create_workspace_row(self) -> QWidget:
        """Create the workspace path display and folder button."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)

        workspace_path = self._coordinator.workspace
        path_label = QLabel(f"<b>Project:</b> {workspace_path}")
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path_label, stretch=1)

        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.setFixedWidth(100)
        layout.addWidget(self._open_folder_btn)

        return row

    def _create_charuco_group(self) -> QGroupBox:
        """Create the charuco board configuration group."""
        group = QGroupBox("Charuco Board Configuration")
        layout = QVBoxLayout(group)

        # Embed the CharucoConfigPanel
        self._charuco_panel = CharucoConfigPanel(self._coordinator.charuco)
        layout.addWidget(self._charuco_panel)

        # Helper text
        helper_label = QLabel("<i>Top left corner is point (0,0,0) when setting capture volume origin</i>")
        helper_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        helper_label.setStyleSheet("color: #888;")
        layout.addWidget(helper_label)

        # PNG save buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 8, 0, 0)

        self._save_png_btn = QPushButton("Save PNG")
        self._save_png_btn.setFixedWidth(100)
        btn_layout.addWidget(self._save_png_btn)

        self._save_mirror_btn = QPushButton("Save Mirror PNG")
        self._save_mirror_btn.setFixedWidth(120)
        btn_layout.addWidget(self._save_mirror_btn)

        btn_layout.addStretch()
        layout.addWidget(btn_row)

        # Charuco preview image
        self._charuco_preview = QLabel()
        self._charuco_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._charuco_preview.setMinimumHeight(200)
        self._charuco_preview.setStyleSheet(
            "QLabel { background-color: #2a2a2a; border: 1px solid #555; border-radius: 4px; }"
        )
        layout.addWidget(self._charuco_preview)

        # Edge length helper text
        edge_helper = QLabel("<i>Printed square size will set the scale of the capture volume</i>")
        edge_helper.setAlignment(Qt.AlignmentFlag.AlignCenter)
        edge_helper.setStyleSheet("color: #888;")
        layout.addWidget(edge_helper)

        return group

    def _create_workflow_group(self) -> QGroupBox:
        """Create the workflow status checklist group."""
        group = QGroupBox("Workflow Status")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        # Camera count display (read-only, derived from filesystem)
        self._camera_count_label = QLabel()
        self._camera_count_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._camera_count_label)

        # Create workflow step rows
        self._intrinsic_row = WorkflowStepRow("Intrinsic Calibration", "Cameras")
        self._intrinsic_row.navigation_requested.connect(self.tab_navigation_requested)
        layout.addWidget(self._intrinsic_row)

        self._extraction_row = WorkflowStepRow("2D Landmark Extraction", "Multi-Camera")
        self._extraction_row.navigation_requested.connect(self.tab_navigation_requested)
        layout.addWidget(self._extraction_row)

        self._extrinsic_row = WorkflowStepRow("Extrinsic Calibration", "Capture Volume")
        self._extrinsic_row.navigation_requested.connect(self.tab_navigation_requested)
        layout.addWidget(self._extrinsic_row)

        self._reconstruction_row = WorkflowStepRow("Reconstruction", "Reconstruction")
        self._reconstruction_row.navigation_requested.connect(self.tab_navigation_requested)
        layout.addWidget(self._reconstruction_row)

        return group

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire up signal connections."""
        # Single signal for status refresh
        self._coordinator.status_changed.connect(self._refresh_status)

        # Charuco panel still needs specific handling for preview update
        self._coordinator.charuco_changed.connect(self._update_charuco_preview)
        self._charuco_panel.config_changed.connect(self._on_panel_config_changed)

        # UI buttons
        self._open_folder_btn.clicked.connect(self._open_workspace_folder)
        self._save_png_btn.clicked.connect(self._save_charuco_png)
        self._save_mirror_btn.clicked.connect(self._save_charuco_mirror_png)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _on_panel_config_changed(self) -> None:
        """Handle config change from CharucoConfigPanel."""
        charuco = self._charuco_panel.get_charuco()
        self._coordinator.update_charuco(charuco)
        # Note: update_charuco emits charuco_changed, which triggers _update_charuco_preview

    def _open_workspace_folder(self) -> None:
        """Open the workspace directory in the system file manager."""
        workspace_path = str(self._coordinator.workspace)
        logger.info(f"Opening workspace folder: {workspace_path}")

        if sys.platform == "win32":
            os.startfile(workspace_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", workspace_path], check=False)
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", workspace_path], check=False)

    def _save_charuco_png(self) -> None:
        """Save the charuco board as a PNG file."""
        default_path = Path(self._coordinator.workspace) / "charuco.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Charuco Board",
            str(default_path),
            "PNG Files (*.png)",
        )

        if file_path:
            charuco = self._charuco_panel.get_charuco()
            charuco.save_image(file_path)
            logger.info(f"Saved charuco board to {file_path}")

    def _save_charuco_mirror_png(self) -> None:
        """Save the charuco board as a mirrored PNG file."""
        default_path = Path(self._coordinator.workspace) / "charuco_mirror.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mirrored Charuco Board",
            str(default_path),
            "PNG Files (*.png)",
        )

        if file_path:
            charuco = self._charuco_panel.get_charuco()
            charuco.save_mirror_image(file_path)
            logger.info(f"Saved mirrored charuco board to {file_path}")

    # -------------------------------------------------------------------------
    # Status Refresh
    # -------------------------------------------------------------------------

    def _refresh_status(self) -> None:
        """Query Coordinator and update status display."""
        status = self._coordinator.get_workflow_status()

        # Update camera count display
        if status.camera_count > 0:
            self._camera_count_label.setText(f"Detected cameras: {status.camera_count} (from extrinsic videos)")
        else:
            self._camera_count_label.setText("No cameras detected (add port_N.mp4 files to calibration/extrinsic/)")

        self._update_intrinsic_row(status)
        self._update_extraction_row(status)
        self._update_extrinsic_row(status)
        self._update_reconstruction_row(status)

        logger.debug("ProjectSetupView status refreshed")

    def _update_intrinsic_row(self, status: WorkflowStatus) -> None:
        """Update the intrinsic calibration status row."""
        step_status = status.intrinsic_step_status

        if step_status == StepStatus.COMPLETE:
            detail = f"{status.camera_count}/{status.camera_count} cameras calibrated"
        elif step_status == StepStatus.INCOMPLETE:
            calibrated_count = status.camera_count - len(status.cameras_needing_calibration)
            detail = f"{calibrated_count}/{status.camera_count} cameras calibrated"
            if status.cameras_needing_calibration:
                ports = ", ".join(str(p) for p in status.cameras_needing_calibration[:3])
                if len(status.cameras_needing_calibration) > 3:
                    ports += "..."
                detail += f" (need: {ports})"
        else:
            if status.intrinsic_videos_missing:
                ports = ", ".join(str(p) for p in status.intrinsic_videos_missing[:3])
                if len(status.intrinsic_videos_missing) > 3:
                    ports += "..."
                detail = f"Missing videos: port {ports}"
            else:
                detail = "Not started"

        self._intrinsic_row.set_status(step_status, detail)

    def _update_extraction_row(self, status: WorkflowStatus) -> None:
        """Update the 2D extraction status row."""
        step_status = status.extrinsic_2d_step_status

        if step_status == StepStatus.COMPLETE:
            detail = "Complete"
        elif step_status == StepStatus.INCOMPLETE:
            detail = "Ready to process"
        else:
            if not status.intrinsic_calibration_complete:
                detail = "Waiting for intrinsic calibration"
            elif status.extrinsic_videos_missing:
                ports = ", ".join(str(p) for p in status.extrinsic_videos_missing[:3])
                if len(status.extrinsic_videos_missing) > 3:
                    ports += "..."
                detail = f"Missing videos: port {ports}"
            else:
                detail = "Not started"

        self._extraction_row.set_status(step_status, detail)

    def _update_extrinsic_row(self, status: WorkflowStatus) -> None:
        """Update the extrinsic calibration status row."""
        step_status = status.extrinsic_calibration_step_status

        if step_status == StepStatus.COMPLETE:
            detail = "Complete"
        elif step_status == StepStatus.INCOMPLETE:
            detail = "Ready to calibrate"
        else:
            if not status.extrinsic_2d_extraction_complete:
                detail = "Waiting for 2D extraction"
            else:
                detail = "Not started"

        self._extrinsic_row.set_status(step_status, detail)

    def _update_reconstruction_row(self, status: WorkflowStatus) -> None:
        """Update the reconstruction status row."""
        if not status.extrinsic_calibration_complete:
            # Prerequisites not met
            detail = "Waiting for extrinsic calibration"
            step_status = StepStatus.NOT_STARTED
        elif status.recordings_available:
            # Capability unlocked - show as AVAILABLE (blue)
            count = len(status.recording_names)
            detail = f"{count} recording{'s' if count != 1 else ''} available"
            step_status = StepStatus.AVAILABLE
        else:
            # Calibrated but no recordings yet
            detail = "No recordings yet"
            step_status = StepStatus.NOT_STARTED

        self._reconstruction_row.set_status(step_status, detail)

    def _update_charuco_preview(self) -> None:
        """Update the charuco board preview image."""
        charuco = self._charuco_panel.get_charuco()

        # Calculate preview dimensions maintaining aspect ratio
        preview_width = 400
        preview_height = 300

        board_width = charuco.board_width
        board_height = charuco.board_height

        if board_height > board_width:
            target_height = preview_height
            target_width = int(target_height * (board_width / board_height))
        else:
            target_width = preview_width
            target_height = int(target_width * (board_height / board_width))

        try:
            pixmap = charuco.board_pixmap(target_width, target_height)
            self._charuco_preview.setPixmap(pixmap)
            self._charuco_preview.setStyleSheet(
                "QLabel { background-color: #2a2a2a; border: 1px solid #555; border-radius: 4px; }"
            )
            self._charuco_preview.setToolTip("")
        except Exception as e:
            logger.error(f"Failed to create charuco preview: {e}")
            self._charuco_preview.setPixmap(QPixmap())
            self._charuco_preview.setText(
                "Unable to create board with current dimensions.\n"
                "The dictionary may be too small or aspect ratio too extreme."
            )
            self._charuco_preview.setStyleSheet(
                "QLabel { color: red; background-color: #2a2a2a; "
                "border: 1px solid #555; border-radius: 4px; padding: 20px; }"
            )
            self._charuco_preview.setToolTip("Try adjusting dimensions to have a less extreme ratio")
