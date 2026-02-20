"""Project setup and workflow status view.

This is the landing tab for Caliscope that:
1. Displays workspace path with folder access
2. Configures intrinsic and extrinsic calibration targets
3. Shows board previews with PNG export
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

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import cv2

from caliscope.core.workflow_status import StepStatus, WorkflowStatus
from caliscope.gui.utils.aruco_preview import render_aruco_pixmap
from caliscope.gui.utils.chessboard_preview import render_chessboard_pixmap
from caliscope.gui.utils.charuco_preview import render_charuco_pixmap
from caliscope.gui.widgets.aruco_target_config_panel import ArucoTargetConfigPanel
from caliscope.gui.widgets.chessboard_config_panel import ChessboardConfigPanel
from caliscope.gui.widgets.charuco_config_panel import CharucoConfigPanel
from caliscope.workspace_coordinator import WorkspaceCoordinator

logger = logging.getLogger(__name__)

# Stacked widget page indices for intrinsic target
_INTRINSIC_PAGE_CHARUCO = 0
_INTRINSIC_PAGE_CHESSBOARD = 1

# Stacked widget page indices for extrinsic target
_EXTRINSIC_PAGE_ARUCO = 0
_EXTRINSIC_PAGE_CHARUCO = 1


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
    - intrinsic_target_changed: Intrinsic target config updated (refresh preview)
    - extrinsic_target_changed: Extrinsic target config updated (refresh preview)
    """

    tab_navigation_requested = Signal(str)  # Tab name to navigate to

    def __init__(self, coordinator: WorkspaceCoordinator) -> None:
        super().__init__()
        self._coordinator = coordinator

        # Panel references (populated during setup)
        self._intrinsic_charuco_panel: CharucoConfigPanel | None = None
        self._intrinsic_chessboard_panel: ChessboardConfigPanel | None = None
        self._extrinsic_aruco_panel: ArucoTargetConfigPanel | None = None
        self._extrinsic_charuco_panel: CharucoConfigPanel | None = None

        self._setup_ui()
        self._connect_signals()
        self._refresh_status()

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
        config_row.addWidget(self._create_intrinsic_target_group())
        config_row.addWidget(self._create_extrinsic_target_group())
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

    def _create_intrinsic_target_group(self) -> QGroupBox:
        """Create the intrinsic calibration target configuration group."""
        group = QGroupBox("Intrinsic Calibration Target")
        main_layout = QVBoxLayout(group)

        # Target type combo box
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Target Type:"))
        self._intrinsic_type_combo = QComboBox()
        self._intrinsic_type_combo.addItem("ChArUco Board (recommended)", "charuco")
        self._intrinsic_type_combo.addItem("Chessboard", "chessboard")
        type_row.addWidget(self._intrinsic_type_combo)
        type_row.addStretch()
        main_layout.addLayout(type_row)

        # Stacked widget for config + preview
        self._intrinsic_stack = QStackedWidget()

        # Page 0: Charuco
        charuco_page = QWidget()
        charuco_layout = QHBoxLayout(charuco_page)
        charuco = self._coordinator.targets_repository.load_intrinsic_charuco()
        self._intrinsic_charuco_panel = CharucoConfigPanel(charuco)
        charuco_layout.addWidget(self._intrinsic_charuco_panel)

        self._intrinsic_charuco_preview = QLabel()
        self._intrinsic_charuco_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._intrinsic_charuco_preview.setMinimumSize(200, 200)
        self._intrinsic_charuco_preview.setStyleSheet(
            "QLabel { background-color: #2a2a2a; border: 1px solid #555; border-radius: 4px; }"
        )
        charuco_layout.addWidget(self._intrinsic_charuco_preview)
        self._intrinsic_stack.addWidget(charuco_page)

        # Page 1: Chessboard
        chessboard_page = QWidget()
        chessboard_layout = QHBoxLayout(chessboard_page)
        chessboard = self._coordinator.targets_repository.load_chessboard()
        self._intrinsic_chessboard_panel = ChessboardConfigPanel(chessboard)
        chessboard_layout.addWidget(self._intrinsic_chessboard_panel)

        self._intrinsic_chessboard_preview = QLabel()
        self._intrinsic_chessboard_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._intrinsic_chessboard_preview.setMinimumSize(200, 200)
        self._intrinsic_chessboard_preview.setStyleSheet(
            "QLabel { background-color: #2a2a2a; border: 1px solid #555; border-radius: 4px; }"
        )
        chessboard_layout.addWidget(self._intrinsic_chessboard_preview)
        self._intrinsic_stack.addWidget(chessboard_page)

        main_layout.addWidget(self._intrinsic_stack)

        # Save button
        self._intrinsic_save_btn = QPushButton()
        main_layout.addWidget(self._intrinsic_save_btn)

        # Set initial state from routing
        routing = self._coordinator.targets_repository.get_routing()
        if routing.intrinsic_target_type == "charuco":
            self._intrinsic_type_combo.setCurrentIndex(0)
            self._intrinsic_stack.setCurrentIndex(_INTRINSIC_PAGE_CHARUCO)
            self._intrinsic_save_btn.setText("Save Board + Mirror")
            self._update_intrinsic_charuco_preview()
        else:
            self._intrinsic_type_combo.setCurrentIndex(1)
            self._intrinsic_stack.setCurrentIndex(_INTRINSIC_PAGE_CHESSBOARD)
            self._intrinsic_save_btn.setText("Save PNG")
            self._update_intrinsic_chessboard_preview()

        return group

    def _create_extrinsic_target_group(self) -> QGroupBox:
        """Create the extrinsic calibration target configuration group."""
        group = QGroupBox("Extrinsic Calibration Target")
        main_layout = QVBoxLayout(group)

        # Target type combo box
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Target Type:"))
        self._extrinsic_type_combo = QComboBox()
        self._extrinsic_type_combo.addItem("ChArUco Board", "charuco")
        self._extrinsic_type_combo.addItem("ArUco Marker", "aruco")
        type_row.addWidget(self._extrinsic_type_combo)
        type_row.addStretch()
        main_layout.addLayout(type_row)

        # Same-as-intrinsic checkbox (only visible when both are charuco)
        self._same_as_intrinsic_check = QCheckBox("Same as intrinsic target")
        main_layout.addWidget(self._same_as_intrinsic_check)

        # Stacked widget for config + preview
        self._extrinsic_stack = QStackedWidget()

        # Page 0: ArUco
        aruco_page = QWidget()
        aruco_layout = QHBoxLayout(aruco_page)
        aruco_target = self._coordinator.targets_repository.load_aruco_target()
        self._extrinsic_aruco_panel = ArucoTargetConfigPanel(aruco_target)
        aruco_layout.addWidget(self._extrinsic_aruco_panel)

        self._extrinsic_aruco_preview = QLabel()
        self._extrinsic_aruco_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._extrinsic_aruco_preview.setMinimumSize(120, 120)
        aruco_layout.addWidget(self._extrinsic_aruco_preview, stretch=1)
        self._extrinsic_stack.addWidget(aruco_page)

        # Page 1: Charuco (can be disabled or enabled based on same-as-intrinsic)
        charuco_page = QWidget()
        charuco_layout = QHBoxLayout(charuco_page)
        extrinsic_charuco = self._coordinator.targets_repository.load_extrinsic_charuco()
        self._extrinsic_charuco_panel = CharucoConfigPanel(extrinsic_charuco)
        charuco_layout.addWidget(self._extrinsic_charuco_panel)

        self._extrinsic_charuco_preview = QLabel()
        self._extrinsic_charuco_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._extrinsic_charuco_preview.setMinimumSize(200, 200)
        self._extrinsic_charuco_preview.setStyleSheet(
            "QLabel { background-color: #2a2a2a; border: 1px solid #555; border-radius: 4px; }"
        )
        charuco_layout.addWidget(self._extrinsic_charuco_preview)
        self._extrinsic_stack.addWidget(charuco_page)

        main_layout.addWidget(self._extrinsic_stack)

        # Save button
        self._extrinsic_save_btn = QPushButton()
        main_layout.addWidget(self._extrinsic_save_btn)

        # Set initial state from routing
        routing = self._coordinator.targets_repository.get_routing()
        if routing.extrinsic_target_type == "charuco":
            self._extrinsic_type_combo.setCurrentIndex(0)
            self._same_as_intrinsic_check.setChecked(routing.extrinsic_charuco_same_as_intrinsic)
            # Visibility of checkbox
            intrinsic_is_charuco = routing.intrinsic_target_type == "charuco"
            self._same_as_intrinsic_check.setVisible(intrinsic_is_charuco)
            self._update_extrinsic_stack()
        else:
            self._extrinsic_type_combo.setCurrentIndex(1)
            self._same_as_intrinsic_check.setVisible(False)
            self._extrinsic_stack.setCurrentIndex(_EXTRINSIC_PAGE_ARUCO)
            self._extrinsic_save_btn.setText("Save PNG")
            self._update_extrinsic_aruco_preview()

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
        # Status refresh
        self._coordinator.status_changed.connect(self._refresh_status)

        # Target config panels -> coordinator
        if self._intrinsic_charuco_panel is not None:
            self._intrinsic_charuco_panel.config_changed.connect(self._on_intrinsic_charuco_changed)
        if self._intrinsic_chessboard_panel is not None:
            self._intrinsic_chessboard_panel.config_changed.connect(self._on_intrinsic_chessboard_changed)
        if self._extrinsic_aruco_panel is not None:
            self._extrinsic_aruco_panel.config_changed.connect(self._on_extrinsic_aruco_changed)
        if self._extrinsic_charuco_panel is not None:
            self._extrinsic_charuco_panel.config_changed.connect(self._on_extrinsic_charuco_changed)

        # Combo boxes
        self._intrinsic_type_combo.currentIndexChanged.connect(self._on_intrinsic_type_changed)
        self._extrinsic_type_combo.currentIndexChanged.connect(self._on_extrinsic_type_changed)
        self._same_as_intrinsic_check.toggled.connect(self._on_same_as_intrinsic_changed)

        # Coordinator -> view (for preview refresh when config changes externally)
        self._coordinator.intrinsic_target_changed.connect(self._on_intrinsic_target_changed)
        self._coordinator.extrinsic_target_changed.connect(self._on_extrinsic_target_changed)

        # UI buttons
        self._open_folder_btn.clicked.connect(self._open_workspace_folder)
        self._intrinsic_save_btn.clicked.connect(self._save_intrinsic_target)
        self._extrinsic_save_btn.clicked.connect(self._save_extrinsic_target)

    # -------------------------------------------------------------------------
    # Event Handlers - Intrinsic Target
    # -------------------------------------------------------------------------

    def _on_intrinsic_type_changed(self, index: int) -> None:
        """Handle intrinsic target type combo box change."""
        target_type = self._intrinsic_type_combo.currentData()
        if target_type == "charuco":
            self._intrinsic_stack.setCurrentIndex(_INTRINSIC_PAGE_CHARUCO)
        else:
            self._intrinsic_stack.setCurrentIndex(_INTRINSIC_PAGE_CHESSBOARD)
        self._coordinator.update_intrinsic_target_type(target_type)

        # Update save button text
        if target_type == "charuco":
            self._intrinsic_save_btn.setText("Save Board + Mirror")
        else:
            self._intrinsic_save_btn.setText("Save PNG")

        # If switching away from charuco while extrinsic same-as-intrinsic is checked,
        # uncheck it and switch extrinsic to editable mode
        routing = self._coordinator.targets_repository.get_routing()
        if (
            target_type != "charuco"
            and routing.extrinsic_target_type == "charuco"
            and routing.extrinsic_charuco_same_as_intrinsic
        ):
            self._same_as_intrinsic_check.setChecked(False)

        # Update checkbox visibility
        self._same_as_intrinsic_check.setVisible(
            target_type == "charuco" and routing.extrinsic_target_type == "charuco"
        )

    def _on_intrinsic_charuco_changed(self) -> None:
        """Handle intrinsic charuco config panel change."""
        if self._intrinsic_charuco_panel is None:
            return
        charuco = self._intrinsic_charuco_panel.get_charuco()
        self._coordinator.update_intrinsic_charuco(charuco)

    def _on_intrinsic_chessboard_changed(self) -> None:
        """Handle intrinsic chessboard config panel change."""
        if self._intrinsic_chessboard_panel is None:
            return
        chessboard = self._intrinsic_chessboard_panel.get_chessboard()
        self._coordinator.update_intrinsic_chessboard(chessboard)

    def _on_intrinsic_target_changed(self) -> None:
        """Refresh intrinsic preview and sync extrinsic panel if needed."""
        target_type = self._coordinator.targets_repository.intrinsic_target_type
        if target_type == "charuco":
            self._update_intrinsic_charuco_preview()
        else:
            self._update_intrinsic_chessboard_preview()

        # If extrinsic is same-as-intrinsic, update extrinsic charuco panel + preview
        routing = self._coordinator.targets_repository.get_routing()
        if routing.extrinsic_target_type == "charuco" and routing.extrinsic_charuco_same_as_intrinsic:
            charuco = self._coordinator.targets_repository.load_intrinsic_charuco()
            if self._extrinsic_charuco_panel is not None:
                self._extrinsic_charuco_panel.set_values(charuco)
            self._update_extrinsic_charuco_preview()

    def _save_intrinsic_target(self) -> None:
        """Save intrinsic target board image(s) to file."""
        target_type = self._coordinator.targets_repository.intrinsic_target_type
        if target_type == "charuco":
            self._save_charuco_images(self._intrinsic_charuco_panel)
        else:
            self._save_chessboard_png(self._intrinsic_chessboard_panel)

    # -------------------------------------------------------------------------
    # Event Handlers - Extrinsic Target
    # -------------------------------------------------------------------------

    def _on_extrinsic_type_changed(self, index: int) -> None:
        """Handle extrinsic target type combo box change."""
        target_type = self._extrinsic_type_combo.currentData()
        self._coordinator.update_extrinsic_target_type(target_type)

        # Show/hide same_as_intrinsic checkbox
        routing = self._coordinator.targets_repository.get_routing()
        intrinsic_is_charuco = routing.intrinsic_target_type == "charuco"
        self._same_as_intrinsic_check.setVisible(target_type == "charuco" and intrinsic_is_charuco)

        self._update_extrinsic_stack()

    def _on_same_as_intrinsic_changed(self, checked: bool) -> None:
        """Handle same-as-intrinsic checkbox change."""
        self._coordinator.set_extrinsic_charuco_same_as_intrinsic(checked)
        self._update_extrinsic_stack()

    def _update_extrinsic_stack(self) -> None:
        """Set the correct stacked widget page based on current state."""
        target_type = self._extrinsic_type_combo.currentData()
        if target_type == "aruco":
            self._extrinsic_stack.setCurrentIndex(_EXTRINSIC_PAGE_ARUCO)
            self._extrinsic_save_btn.setText("Save PNG")
            self._extrinsic_save_btn.setEnabled(True)
            self._extrinsic_save_btn.setToolTip("")
            self._update_extrinsic_aruco_preview()
        else:  # "charuco"
            same_as_intrinsic = self._same_as_intrinsic_check.isChecked()
            self._extrinsic_stack.setCurrentIndex(_EXTRINSIC_PAGE_CHARUCO)
            self._extrinsic_save_btn.setText("Save Board + Mirror")

            if same_as_intrinsic:
                # Disable save button and sync panel from intrinsic
                self._extrinsic_save_btn.setEnabled(False)
                self._extrinsic_save_btn.setToolTip(
                    "Board images are the same as the intrinsic target — use the intrinsic Save button."
                )
                charuco = self._coordinator.targets_repository.load_intrinsic_charuco()
                if self._extrinsic_charuco_panel is not None:
                    self._extrinsic_charuco_panel.setEnabled(False)
                    self._extrinsic_charuco_panel.set_values(charuco)
            else:
                # Enable save button and editable panel
                self._extrinsic_save_btn.setEnabled(True)
                self._extrinsic_save_btn.setToolTip("")
                if self._extrinsic_charuco_panel is not None:
                    self._extrinsic_charuco_panel.setEnabled(True)

            self._update_extrinsic_charuco_preview()

    def _on_extrinsic_aruco_changed(self) -> None:
        """Handle extrinsic ArUco config panel change."""
        if self._extrinsic_aruco_panel is None:
            return
        target = self._extrinsic_aruco_panel.get_aruco_target()
        self._coordinator.update_extrinsic_aruco_target(target)

    def _on_extrinsic_charuco_changed(self) -> None:
        """Handle extrinsic charuco config panel change."""
        if self._extrinsic_charuco_panel is None:
            return
        charuco = self._extrinsic_charuco_panel.get_charuco()
        self._coordinator.update_extrinsic_charuco(charuco)

    def _on_extrinsic_target_changed(self) -> None:
        """Refresh extrinsic preview."""
        target_type = self._coordinator.targets_repository.extrinsic_target_type
        if target_type == "aruco":
            self._update_extrinsic_aruco_preview()
        else:
            self._update_extrinsic_charuco_preview()

    def _save_extrinsic_target(self) -> None:
        """Save extrinsic target board image(s) to file."""
        target_type = self._coordinator.targets_repository.extrinsic_target_type
        if target_type == "aruco":
            self._save_aruco_png(self._extrinsic_aruco_panel)
        else:
            self._save_charuco_images(self._extrinsic_charuco_panel)

    # -------------------------------------------------------------------------
    # Preview Updates
    # -------------------------------------------------------------------------

    def _update_intrinsic_charuco_preview(self) -> None:
        """Update intrinsic charuco preview."""
        charuco = self._coordinator.targets_repository.load_intrinsic_charuco()
        pixmap = render_charuco_pixmap(charuco, 200)
        self._intrinsic_charuco_preview.setPixmap(pixmap)

    def _update_intrinsic_chessboard_preview(self) -> None:
        """Update intrinsic chessboard preview."""
        chessboard = self._coordinator.targets_repository.load_chessboard()
        pixmap = render_chessboard_pixmap(chessboard, 200)
        self._intrinsic_chessboard_preview.setPixmap(pixmap)

    def _update_extrinsic_charuco_preview(self) -> None:
        """Update extrinsic charuco preview."""
        charuco = self._coordinator.targets_repository.load_extrinsic_charuco()
        pixmap = render_charuco_pixmap(charuco, 200)
        self._extrinsic_charuco_preview.setPixmap(pixmap)

    def _update_extrinsic_aruco_preview(self) -> None:
        """Update extrinsic ArUco preview."""
        target = self._coordinator.targets_repository.load_aruco_target()
        marker_id = target.marker_ids[0] if target.marker_ids else 0
        pixmap = render_aruco_pixmap(target, marker_id, 120)
        self._extrinsic_aruco_preview.setPixmap(pixmap)

    # -------------------------------------------------------------------------
    # File Save Handlers
    # -------------------------------------------------------------------------

    def _save_charuco_images(self, panel: CharucoConfigPanel | None) -> None:
        """Save charuco board as front PNG + mirror PNG."""
        if panel is None:
            return
        charuco = panel.get_charuco()
        default_dir = self._coordinator.workspace

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ChArUco Board", str(default_dir / "charuco_board.png"), "PNG Files (*.png)"
        )
        if file_path:
            charuco.save_image(file_path)
            # Save mirror alongside with _mirror suffix
            p = Path(file_path)
            mirror_path = p.parent / f"{p.stem}_mirror{p.suffix}"
            charuco.save_mirror_image(str(mirror_path))
            logger.info(f"Saved charuco board to {file_path} and mirror to {mirror_path}")

    def _save_chessboard_png(self, panel: ChessboardConfigPanel | None) -> None:
        """Save chessboard as a high-resolution PNG file."""
        if panel is None:
            return
        chessboard = panel.get_chessboard()
        default_path = Path(self._coordinator.workspace) / "chessboard.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chessboard",
            str(default_path),
            "PNG Files (*.png)",
        )
        if file_path:
            pixmap = render_chessboard_pixmap(chessboard, 2000)
            pixmap.save(file_path, "PNG")
            logger.info(f"Saved chessboard to {file_path}")

    def _save_aruco_png(self, panel: ArucoTargetConfigPanel | None) -> None:
        """Save ArUco marker image to file."""
        if panel is None:
            return
        target = panel.get_aruco_target()
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
    # Other Handlers
    # -------------------------------------------------------------------------

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
            self._camera_count_label.setText("No cameras detected (add cam_N.mp4 files to calibration/extrinsic/)")

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
                cam_labels = ", ".join(str(p) for p in status.cameras_needing_calibration[:3])
                if len(status.cameras_needing_calibration) > 3:
                    cam_labels += "..."
                detail += f" (need: {cam_labels})"
        else:
            if status.intrinsic_videos_missing:
                cam_labels = ", ".join(str(p) for p in status.intrinsic_videos_missing[:3])
                if len(status.intrinsic_videos_missing) > 3:
                    cam_labels += "..."
                detail = f"Missing videos: cam {cam_labels}"
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
                cam_labels = ", ".join(str(p) for p in status.extrinsic_videos_missing[:3])
                if len(status.extrinsic_videos_missing) > 3:
                    cam_labels += "..."
                detail = f"Missing videos: cam {cam_labels}"
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
