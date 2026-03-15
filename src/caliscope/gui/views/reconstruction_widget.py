"""Widget for reconstruction workflow (post-processing).

Provides UI for selecting recordings, choosing trackers, and running
reconstruction to generate 3D trajectories. Visualization via Qt3D
shows triangulated points when reconstruction completes.

This is a thin MVP widget following the state-driven UI pattern.
"""

import logging

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from caliscope.gui.presenters.reconstruction_presenter import (
    ReconstructionPresenter,
    ReconstructionState,
)
from caliscope.gui.view_models.playback_view_model import PlaybackViewModel
from caliscope.gui.widgets.qt3d_playback_widget import Qt3DPlaybackWidget
from caliscope import MODELS_DIR
from caliscope.gui.theme import Colors
from caliscope.trackers import tracker_registry

logger = logging.getLogger(__name__)


class ReconstructionWidget(QWidget):
    """Widget for post-processing reconstruction workflow.

    Connects to ReconstructionPresenter for business logic and state management.
    UI updates derive from presenter state via _update_ui_for_state().
    """

    def __init__(
        self,
        presenter: ReconstructionPresenter,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._presenter = presenter
        self._viz_widget: Qt3DPlaybackWidget | None = None
        self._viz_pending = False  # Debounce flag for _update_visualization

        self._setup_ui()
        self._connect_signals()
        self._populate_initial_data()

        # Initial UI state
        self._update_ui_for_state(presenter.state)

    def _setup_ui(self) -> None:
        """Create UI elements with left panel controls and right panel visualization."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Use splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # === Left Panel (Controls) ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)

        # Recording selection group
        recording_group = QGroupBox("Recording")
        recording_layout = QVBoxLayout(recording_group)

        self._recording_list = QListWidget()
        self._recording_list.setMaximumHeight(150)
        recording_layout.addWidget(self._recording_list)

        left_layout.addWidget(recording_group)

        # Tracker selection group
        tracker_group = QGroupBox("Tracker")
        tracker_layout = QVBoxLayout(tracker_group)

        self._tracker_combo = QComboBox()
        tracker_layout.addWidget(self._tracker_combo)

        self._models_folder_link = QLabel(
            f'<a href="file://{MODELS_DIR}" style="color: {Colors.PRIMARY};">Open Models Folder</a>'
        )
        self._models_folder_link.setStyleSheet("font-size: 11px;")
        self._models_folder_link.setCursor(Qt.CursorShape.PointingHandCursor)
        tracker_layout.addWidget(self._models_folder_link)

        left_layout.addWidget(tracker_group)

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self._state_label = QLabel("State: IDLE")
        status_layout.addWidget(self._state_label)

        self._status_message = QLabel("Select a recording to begin")
        self._status_message.setWordWrap(True)
        status_layout.addWidget(self._status_message)

        self._calibration_indicator = QLabel()
        self._calibration_indicator.setStyleSheet("color: #888; font-style: italic;")
        self._calibration_indicator.setWordWrap(True)
        self._calibration_indicator.hide()
        status_layout.addWidget(self._calibration_indicator)

        left_layout.addWidget(status_group)

        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self._process_btn = QPushButton("Process")
        self._process_btn.setEnabled(False)
        actions_layout.addWidget(self._process_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        actions_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setWordWrap(True)
        self._progress_label.hide()
        actions_layout.addWidget(self._progress_label)

        self._open_output_btn = QPushButton("Open Output Folder")
        self._open_output_btn.setToolTip("Open folder containing xyz/TRC output files")
        self._open_output_btn.setEnabled(False)
        self._open_output_btn.hide()
        actions_layout.addWidget(self._open_output_btn)

        left_layout.addWidget(actions_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # === Right Panel (Visualization) ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Container for 3D visualization widget (progressive enhancement: always show cameras)
        self._viz_container = QVBoxLayout()
        right_layout.addLayout(self._viz_container)

        splitter.addWidget(right_panel)

        # Set splitter proportions (1:3 ratio for left:right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def _connect_signals(self) -> None:
        """Connect presenter signals and UI events."""
        # Presenter → View
        self._presenter.state_changed.connect(self._update_ui_for_state)
        self._presenter.progress_updated.connect(self._update_progress)
        self._presenter.reconstruction_complete.connect(self._on_reconstruction_complete)
        self._presenter.reconstruction_failed.connect(self._on_reconstruction_failed)

        # View → Presenter (via adapters)
        self._recording_list.currentTextChanged.connect(self._on_recording_changed)
        self._tracker_combo.currentIndexChanged.connect(self._on_tracker_changed)
        self._process_btn.clicked.connect(self._on_process_clicked)
        self._open_output_btn.clicked.connect(self._on_open_output_clicked)
        self._presenter.model_download_needed.connect(self._show_model_download_dialog)
        self._models_folder_link.linkActivated.connect(self._on_open_models_folder)

    def _populate_initial_data(self) -> None:
        """Populate lists with available recordings and trackers."""
        # Populate recordings
        recordings = self._presenter.available_recordings
        self._recording_list.clear()
        self._recording_list.addItems(recordings)

        # Auto-select first recording if available
        if recordings:
            self._recording_list.setCurrentRow(0)

        # Populate trackers
        trackers = self._presenter.available_trackers
        self._tracker_combo.clear()
        for tracker_name in trackers:
            display = tracker_registry.display_name_for(tracker_name)
            if not tracker_registry.is_model_ready(tracker_name):
                display += "  (download required)"
            self._tracker_combo.addItem(display, tracker_name)

        # No auto-selection for tracker - user should consciously choose

    def _on_recording_changed(self, name: str) -> None:
        """Handle recording selection change."""
        if name:  # Guard against empty string when list cleared
            self._presenter.select_recording(name)
            self._update_visualization()

    def _on_tracker_changed(self, index: int) -> None:
        """Handle tracker selection change."""
        if index >= 0:
            tracker_name = self._tracker_combo.itemData(index)
            self._presenter.select_tracker(tracker_name)
            self._update_visualization()

    def _on_process_clicked(self) -> None:
        """Handle process button click - action depends on state."""
        state = self._presenter.state
        if state == ReconstructionState.RECONSTRUCTING:
            self._presenter.cancel_reconstruction()
        else:
            # IDLE, COMPLETE, or ERROR - start/restart processing
            self._presenter.start_reconstruction()

    def _on_open_output_clicked(self) -> None:
        """Open the output folder containing xyz/TRC files."""
        output_path = self._presenter.xyz_output_path
        if output_path and output_path.exists():
            # Open the parent directory (tracker output folder)
            folder = output_path.parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _selected_tracker_needs_download(self) -> bool:
        """Check if the currently selected tracker requires a model download.

        Returns False for built-in trackers and ONNX trackers with weights on disk.
        Returns False when no tracker is selected.
        """
        tracker = self._presenter.selected_tracker
        if tracker is None:
            return False
        return not tracker_registry.is_model_ready(tracker)

    def _process_button_text_for_state(self, state: ReconstructionState) -> str:
        """Determine Process button text based on state and model readiness.

        When the selected ONNX tracker needs downloading, the button says
        "Download Model" instead of "Process" so the user knows what will
        happen on click. For all other states the normal text applies.
        """
        if state == ReconstructionState.RECONSTRUCTING:
            return "Cancel"
        if state == ReconstructionState.COMPLETE:
            return "Reprocess"
        if state == ReconstructionState.ERROR:
            return "Retry"

        # IDLE: check if the selected tracker needs a download
        if self._selected_tracker_needs_download():
            tracker = self._presenter.selected_tracker
            if tracker and tracker_registry.has_source_url(tracker):
                return "Download Model"
            return "Model Missing"

        return "Process"

    def _update_ui_for_state(self, state: ReconstructionState) -> None:
        """Update all UI elements based on presenter state.

        Single handler that derives entire UI from current state - prevents
        state/UI divergence.
        """
        # State label
        self._state_label.setText(f"State: {state.name}")

        # State-specific styling
        if state == ReconstructionState.COMPLETE:
            self._state_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif state == ReconstructionState.ERROR:
            self._state_label.setStyleSheet("color: #F44336; font-weight: bold;")
        else:
            self._state_label.setStyleSheet("")

        # Status message
        if state == ReconstructionState.IDLE:
            if self._presenter.selected_recording and self._presenter.selected_tracker:
                if self._selected_tracker_needs_download():
                    tracker = self._presenter.selected_tracker
                    if tracker and tracker_registry.has_source_url(tracker):
                        self._status_message.setText("Model download required before processing")
                    else:
                        self._status_message.setText("Place .onnx file in models folder")
                else:
                    self._status_message.setText("Ready to process")
            elif self._presenter.selected_recording:
                self._status_message.setText("Select a tracker")
            else:
                self._status_message.setText("Select a recording to begin")
        elif state == ReconstructionState.RECONSTRUCTING:
            self._status_message.setText("Processing...")
        elif state == ReconstructionState.COMPLETE:
            self._status_message.setText("Reconstruction complete")
        elif state == ReconstructionState.ERROR:
            error = self._presenter.last_error or "Unknown error"
            self._status_message.setText(f"Error: {error}")

        # Calibration indicator (show when viewing historical data)
        if self._presenter.is_showing_historical_calibration:
            self._calibration_indicator.setText("Showing cameras from this recording's calibration")
            self._calibration_indicator.show()
        else:
            self._calibration_indicator.hide()

        # Process button
        can_process = self._presenter.selected_recording is not None and self._presenter.selected_tracker is not None

        button_text = self._process_button_text_for_state(state)
        self._process_btn.setText(button_text)
        if state == ReconstructionState.RECONSTRUCTING:
            self._process_btn.setEnabled(True)
        elif button_text == "Model Missing":
            self._process_btn.setEnabled(False)
        else:
            self._process_btn.setEnabled(can_process)

        # Progress bar visibility
        if state == ReconstructionState.RECONSTRUCTING:
            self._progress_bar.show()
            self._progress_label.show()
            self._open_output_btn.hide()
        else:
            self._progress_bar.hide()
            self._progress_label.hide()
            self._progress_bar.setValue(0)

        # Open Output button - only visible and enabled in COMPLETE state
        if state == ReconstructionState.COMPLETE:
            self._open_output_btn.show()
            self._open_output_btn.setEnabled(True)
        else:
            self._open_output_btn.hide()
            self._open_output_btn.setEnabled(False)

        # Input controls enabled/disabled
        inputs_enabled = state != ReconstructionState.RECONSTRUCTING
        self._recording_list.setEnabled(inputs_enabled)
        self._tracker_combo.setEnabled(inputs_enabled)

        # Update visualization
        self._update_visualization()

    def _update_progress(self, percent: int, message: str) -> None:
        """Update progress bar and label."""
        self._progress_bar.setValue(percent)
        self._progress_label.setText(message)

    def _on_reconstruction_complete(self, output_path) -> None:
        """Handle successful reconstruction - update visualization."""
        logger.info(f"Reconstruction complete: {output_path}")
        self._update_visualization()

    def _on_reconstruction_failed(self, error: str) -> None:
        """Handle reconstruction failure."""
        logger.error(f"Reconstruction failed: {error}")
        # State change will update UI via _update_ui_for_state

    def _update_visualization(self) -> None:
        """Schedule a visualization update on the next event loop cycle.

        Multiple callers (recording change, tracker change, state change) may
        trigger this in quick succession during init. Debouncing via
        QTimer.singleShot(0) coalesces them into a single scene rebuild,
        avoiding redundant Qt3D scene graph construction.
        """
        if not self._viz_pending:
            self._viz_pending = True
            QTimer.singleShot(0, self._do_update_visualization)

    def _do_update_visualization(self) -> None:
        """Actually rebuild the visualization. Called from debounce timer."""
        self._viz_pending = False
        camera_array = self._presenter.camera_array
        output_path = self._presenter.xyz_output_path

        # Determine what data we have
        try:
            if output_path and output_path.exists():
                # Full data available - load from xyz
                view_model = PlaybackViewModel.from_xyz_csv(
                    xyz_path=str(output_path),
                    camera_array=camera_array,
                    wireframe_segments=self._presenter.wireframe_segments,
                    fps=30,
                )
            else:
                # Camera-only mode - show frustums without points
                view_model = PlaybackViewModel.from_camera_array_only(camera_array)

            # Create or update widget
            if self._viz_widget is None:
                self._viz_widget = Qt3DPlaybackWidget(
                    view_model,
                    camera_size_multiplier=self._presenter.get_camera_size_multiplier(),
                    grid_size_multiplier=self._presenter.get_grid_size_multiplier(),
                )
                self._viz_widget.camera_size_multiplier_changed.connect(self._presenter.save_camera_size_multiplier)
                self._viz_widget.grid_size_multiplier_changed.connect(self._presenter.save_grid_size_multiplier)
                self._viz_container.addWidget(self._viz_widget)
            else:
                self._viz_widget.set_view_model(view_model)

            self._viz_widget.show()

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            # On error, just log — don't break the UI

    def _show_model_download_dialog(self, card: object) -> None:
        """Show the model download dialog when weights are missing."""
        from caliscope.gui.widgets.model_download_dialog import ModelDownloadDialog
        from caliscope.trackers.model_card import ModelCard

        if not isinstance(card, ModelCard):
            logger.error("Expected ModelCard, got %s", type(card))
            return
        dialog = ModelDownloadDialog(card, MODELS_DIR, self._presenter.task_manager, parent=self)
        dialog.finished.connect(self._refresh_tracker_combo_annotations)
        dialog.exec()

    def _refresh_tracker_combo_annotations(self) -> None:
        """Update combo box text and button to reflect current model readiness.

        Re-checks is_model_ready() for each item. Removes or adds
        "(download required)" suffix based on live filesystem state.
        Also updates the process button text in case the user downloaded
        the model while the dialog was open.
        """
        for i in range(self._tracker_combo.count()):
            key = self._tracker_combo.itemData(i)
            if key is None:
                continue
            display = tracker_registry.display_name_for(key)
            if not tracker_registry.is_model_ready(key):
                display += "  (download required)"
            self._tracker_combo.setItemText(i, display)

        # Update button text and status message to reflect new readiness state
        self._update_ui_for_state(self._presenter.state)

    def _on_open_models_folder(self, link: str) -> None:
        """Open MODELS_DIR in the system file manager."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(MODELS_DIR)))

    def cleanup(self) -> None:
        """Explicit cleanup - call before destruction."""
        if self._viz_widget is not None:
            self._viz_widget.close()
            self._viz_widget = None

    def suspend_vtk(self) -> None:
        """Pause VTK rendering when widget is not active."""
        if self._viz_widget is not None:
            self._viz_widget.suspend_vtk()

    def resume_vtk(self) -> None:
        """Resume VTK rendering when widget becomes active."""
        if self._viz_widget is not None:
            self._viz_widget.resume_vtk()

    def closeEvent(self, event) -> None:
        """Handle close event."""
        self.cleanup()
        super().closeEvent(event)
