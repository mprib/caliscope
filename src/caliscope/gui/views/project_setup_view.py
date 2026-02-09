"""Project setup and workflow status view.

This is the landing tab for Caliscope that:
1. Displays workspace path with folder access
2. Configures chessboard for intrinsic calibration
3. Shows chessboard preview with PNG export
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

from PySide6.QtCore import QByteArray, QEvent, QObject, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import cv2

from caliscope.core.aruco_target import ArucoTarget
from caliscope.core.chessboard import Chessboard
from caliscope.core.workflow_status import StepStatus, WorkflowStatus
from caliscope.gui.utils.aruco_preview import render_aruco_pixmap
from caliscope.gui.utils.chessboard_preview import render_chessboard_pixmap
from caliscope.gui.widgets.aruco_target_config_panel import ArucoTargetConfigPanel
from caliscope.gui.widgets.chessboard_config_panel import ChessboardConfigPanel
from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)


class WorkflowStepRow(QWidget):
    """A single row in the workflow status checklist.

    Shows: [SVG icon] Step Name         [status text] [Go to Tab]
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
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # SVG status icon
        self._status_icon = QSvgWidget()
        self._status_icon.setFixedSize(20, 20)
        layout.addWidget(self._status_icon)

        # Step name label
        self._name_label = QLabel(self._step_name)
        self._name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._name_label)

        # Status detail text
        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: #888;")
        layout.addWidget(self._status_label)

        layout.addStretch()

        # Navigation button
        self._nav_btn = QPushButton("Go to Tab")
        self._nav_btn.setFixedWidth(90)
        layout.addWidget(self._nav_btn)

    def _connect_signals(self) -> None:
        self._nav_btn.clicked.connect(lambda: self.navigation_requested.emit(self._target_tab))

    def _load_colored_svg(self, svg_path: Path, color: str) -> None:
        """Load an SVG file and replace currentColor with the specified color.

        Args:
            svg_path: Path to the SVG file
            color: Hex color code (e.g., "#4CAF50")
        """
        with open(svg_path) as f:
            svg_content = f.read()
        # Replace currentColor with the actual color
        colored_svg = svg_content.replace("currentColor", color)
        self._status_icon.load(QByteArray(colored_svg.encode()))

    def set_status(self, status: StepStatus, detail_text: str) -> None:
        """Update the row's visual state.

        Args:
            status: The step's completion status
            detail_text: Descriptive text (e.g., "4/4 cameras calibrated")
        """
        self._status_label.setText(detail_text)

        # Get icons directory
        icons_dir = Path(__file__).parent.parent / "icons"

        # Load appropriate SVG icon with color
        if status == StepStatus.COMPLETE:
            # Green checkmark
            self._load_colored_svg(icons_dir / "status-complete.svg", "#4CAF50")
            self._name_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif status == StepStatus.INCOMPLETE:
            # Yellow/amber partial/in-progress
            self._load_colored_svg(icons_dir / "status-incomplete.svg", "#FFA000")
            self._name_label.setStyleSheet("font-weight: bold; color: #FFA000;")
        elif status == StepStatus.AVAILABLE:
            # Blue indicator for optional capability
            self._load_colored_svg(icons_dir / "status-available.svg", "#2196F3")
            self._name_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        else:  # NOT_STARTED
            # Gray not started
            self._load_colored_svg(icons_dir / "status-not-started.svg", "#666666")
            self._name_label.setStyleSheet("font-weight: bold; color: #888;")


class ProjectSetupView(QWidget):
    """Project setup and status view (no Presenter needed).

    This view wires directly to the Coordinator because it only observes
    workflow status - it doesn't orchestrate any workflows. The status
    is recomputed on each refresh from ground truth in the Coordinator.

    Signal subscriptions:
    - status_changed: Workflow state may have changed (single refresh trigger)
    - chessboard_changed: Board config updated (requires preview refresh)
    """

    tab_navigation_requested = Signal(str)  # Tab name to navigate to

    def __init__(self, coordinator: WorkspaceCoordinator) -> None:
        super().__init__()
        self._coordinator = coordinator

        self._setup_ui()
        self._connect_signals()
        self._refresh_status()
        self._update_chessboard_preview()
        self._update_aruco_preview()

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

        # Calibration target configuration (side by side)
        config_row = QHBoxLayout()
        self._chessboard_group = self._create_chessboard_group()
        config_row.addWidget(self._chessboard_group)
        self._aruco_group = self._create_aruco_target_group()
        config_row.addWidget(self._aruco_group)
        main_layout.addLayout(config_row)

        # Visual separator
        main_layout.addWidget(self._create_separator())

        # Workflow status group
        main_layout.addWidget(self._create_workflow_group())

        main_layout.addStretch()

        scroll.setWidget(content)

        # Set scroll area as main layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _create_separator(self) -> QFrame:
        """Create a horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

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

    def _create_chessboard_group(self) -> QGroupBox:
        """Create the chessboard configuration group."""
        group = QGroupBox("Chessboard Configuration")
        main_layout = QHBoxLayout(group)

        # Left side: config panel
        left_layout = QVBoxLayout()

        # Load initial chessboard from repository or persist default
        if self._coordinator.chessboard_repository.exists():
            initial_chessboard = self._coordinator.chessboard_repository.load()
        else:
            initial_chessboard = Chessboard(rows=6, columns=9)
            self._coordinator.update_chessboard(initial_chessboard)

        self._chessboard_panel = ChessboardConfigPanel(initial_chessboard)
        left_layout.addWidget(self._chessboard_panel)
        main_layout.addLayout(left_layout, stretch=1)

        # Right side: preview + save button
        right_layout = QVBoxLayout()
        right_layout.addStretch()

        self._chessboard_preview = QLabel()
        self._chessboard_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chessboard_preview.setMinimumSize(200, 200)
        self._chessboard_preview.setMaximumSize(550, 550)
        self._chessboard_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._chessboard_preview.setStyleSheet(
            "QLabel { background-color: #2a2a2a; border: 1px solid #555; border-radius: 4px; }"
        )
        right_layout.addWidget(self._chessboard_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        # Single save button (no mirror needed)
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 8, 0, 0)
        self._save_png_btn = QPushButton("Save PNG")
        self._save_png_btn.setFixedWidth(100)
        btn_layout.addWidget(self._save_png_btn)
        right_layout.addWidget(btn_row, alignment=Qt.AlignmentFlag.AlignCenter)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, stretch=1)

        return group

    def _create_aruco_target_group(self) -> QGroupBox:
        """Create ArUco target configuration group for extrinsic calibration.

        Layout: config panel (left) + preview (right) + Save PNG button (below)
        """
        group = QGroupBox("ArUco Target (Extrinsic Calibration)")

        main_layout = QVBoxLayout(group)

        # Top: config + preview side by side
        top_layout = QHBoxLayout()

        # Left: config panel
        if self._coordinator.aruco_target_repository.exists():
            initial_target = self._coordinator.aruco_target_repository.load()
        else:
            initial_target = ArucoTarget.single_marker()
            # Persist default immediately (matches chessboard pattern)
            self._coordinator.update_aruco_target(initial_target)

        self._aruco_panel = ArucoTargetConfigPanel(initial_target)
        self._aruco_panel.config_changed.connect(self._on_aruco_config_changed)
        top_layout.addWidget(self._aruco_panel)

        # Right: preview
        self._aruco_preview = QLabel()
        self._aruco_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._aruco_preview.setMinimumSize(120, 120)
        top_layout.addWidget(self._aruco_preview, stretch=1)

        main_layout.addLayout(top_layout)

        # Bottom: Save PNG button
        self._save_aruco_btn = QPushButton("Save PNG")
        self._save_aruco_btn.clicked.connect(self._save_aruco_png)
        main_layout.addWidget(self._save_aruco_btn)

        # Initial preview
        self._update_aruco_preview()

        # Install event filter for responsive preview sizing
        group.installEventFilter(self)

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

        # Color key legend
        layout.addSpacing(12)
        legend = QLabel(
            '<span style="color: #4CAF50;">●</span> Complete · '
            '<span style="color: #FFA000;">●</span> In Progress · '
            '<span style="color: #2196F3;">●</span> Available · '
            '<span style="color: #666666;">●</span> Not Started'
        )
        legend.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(legend)

        return group

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire up signal connections."""
        # Single signal for status refresh
        self._coordinator.status_changed.connect(self._refresh_status)

        # Chessboard panel needs specific handling for preview update
        self._coordinator.chessboard_changed.connect(self._update_chessboard_preview)
        self._chessboard_panel.config_changed.connect(self._on_chessboard_config_changed)

        # ArUco panel needs specific handling for preview update
        self._coordinator.aruco_target_changed.connect(self._update_aruco_preview)

        # UI buttons
        self._open_folder_btn.clicked.connect(self._open_workspace_folder)
        self._save_png_btn.clicked.connect(self._save_chessboard_png)

        # Install event filter on the chessboard group to detect resize
        self._chessboard_group.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Handle resize events on config groups to update preview sizes."""
        if event.type() == QEvent.Type.Resize:
            from PySide6.QtCore import QTimer

            if watched is self._chessboard_group:
                QTimer.singleShot(0, self._update_chessboard_preview)
            elif watched is self._aruco_group:
                QTimer.singleShot(0, self._update_aruco_preview)
        return super().eventFilter(watched, event)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _on_chessboard_config_changed(self) -> None:
        """Handle config change from ChessboardConfigPanel."""
        chessboard = self._chessboard_panel.get_chessboard()
        self._coordinator.update_chessboard(chessboard)

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

    def _save_chessboard_png(self) -> None:
        """Save the chessboard as a high-resolution PNG file."""
        default_path = Path(self._coordinator.workspace) / "chessboard.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chessboard",
            str(default_path),
            "PNG Files (*.png)",
        )
        if file_path:
            chessboard = self._chessboard_panel.get_chessboard()
            pixmap = render_chessboard_pixmap(chessboard, 2000)
            pixmap.save(file_path, "PNG")
            logger.info(f"Saved chessboard to {file_path}")

    def _on_aruco_config_changed(self) -> None:
        """Handle ArUco config panel changes."""
        target = self._aruco_panel.get_aruco_target()
        self._coordinator.update_aruco_target(target)

    def _update_aruco_preview(self) -> None:
        """Update ArUco preview from current config."""
        if not self._coordinator.aruco_target_repository.exists():
            self._aruco_preview.clear()
            return

        target = self._coordinator.aruco_target_repository.load()
        marker_id = target.marker_ids[0] if target.marker_ids else 0

        # Use available width for preview
        available = self._aruco_preview.width() or 120
        size = min(available, 200)

        pixmap = render_aruco_pixmap(target, marker_id, size)
        self._aruco_preview.setPixmap(pixmap)

    def _save_aruco_png(self) -> None:
        """Save ArUco marker image to file."""
        target = self._aruco_panel.get_aruco_target()
        marker_id = target.marker_ids[0] if target.marker_ids else 0

        default_path = Path(self._coordinator.workspace) / f"aruco_marker_{marker_id}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ArUco Marker",
            str(default_path),
            "PNG Files (*.png)",
        )

        if file_path:
            # Generate high-resolution marker for printing
            bgr = target.generate_marker_image(marker_id, pixels_per_meter=8000)
            cv2.imwrite(file_path, bgr)
            logger.info(f"Saved ArUco marker to {file_path}")

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

    def _update_chessboard_preview(self) -> None:
        """Update the chessboard preview image."""
        chessboard = self._chessboard_panel.get_chessboard()

        # Responsive sizing based on container
        container_width = self._chessboard_preview.width()
        container_height = self._chessboard_preview.height()
        container_size = min(container_width, container_height)
        max_dimension = max(200, min(container_size - 20, 550))

        try:
            pixmap = render_chessboard_pixmap(chessboard, max_dimension)
            self._chessboard_preview.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"Failed to create chessboard preview: {e}")
            self._chessboard_preview.setPixmap(QPixmap())
            self._chessboard_preview.setText("Unable to create preview")
