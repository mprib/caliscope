"""Main tab widget for the Synthetic Calibration Explorer.

Composes all explorer components:
- StoryboardView: 4-panel synchronized 3D visualization
- CoverageHeatmapWidget: Clickable camera pair matrix
- Preset dropdown and Run button for MVP

Layout uses QSplitter for resizable left sidebar and main visualization area.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from caliscope.synthetic.explorer.presenter import ExplorerPresenter, PipelineResult
from caliscope.synthetic.explorer.widgets import CoverageHeatmapWidget, StoryboardView
from caliscope.synthetic.scenario_config import (
    ScenarioConfig,
    default_ring_scenario,
    occluded_camera_scenario,
    sparse_coverage_scenario,
)
from caliscope.task_manager.task_manager import TaskManager

logger = logging.getLogger(__name__)


# Preset scenarios available in the dropdown
PRESETS: dict[str, ScenarioConfig] = {
    "Default Ring (4 cameras)": default_ring_scenario(),
    "Sparse Coverage": sparse_coverage_scenario(),
    "Occluded Camera": occluded_camera_scenario(),
}


class ExplorerTab(QWidget):
    """Main tab for the Synthetic Calibration Explorer.

    Provides an interactive environment for exploring how synthetic calibration
    scenarios perform through the bootstrap-optimize-align pipeline. Users can
    select preset scenarios, run the pipeline, and visualize results across
    four synchronized 3D panels.

    The tab owns its presenter and is responsible for lifecycle management.
    Call cleanup() before destruction (closeEvent alone is not reliable for tabs).
    """

    def __init__(
        self,
        task_manager: TaskManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._presenter = ExplorerPresenter(task_manager, parent=self)

        self._setup_ui()
        self._connect_signals()

        # Initialize with first preset
        self._on_preset_changed(0)

        logger.info("ExplorerTab created")

    def _setup_ui(self) -> None:
        """Build the UI layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Horizontal splitter: sidebar | main area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left sidebar
        sidebar = self._create_sidebar()
        splitter.addWidget(sidebar)

        # Main visualization area
        main_area = self._create_main_area()
        splitter.addWidget(main_area)

        # Set initial splitter sizes (sidebar ~300px, main area gets the rest)
        splitter.setSizes([300, 900])
        splitter.setStretchFactor(0, 0)  # Sidebar doesn't stretch
        splitter.setStretchFactor(1, 1)  # Main area stretches

    def _create_sidebar(self) -> QWidget:
        """Create the left sidebar with controls."""
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Preset selection group
        preset_group = QGroupBox("Scenario")
        preset_layout = QVBoxLayout(preset_group)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(PRESETS.keys())
        preset_layout.addWidget(self._preset_combo)

        layout.addWidget(preset_group)

        # Run button
        self._run_button = QPushButton("Run Pipeline")
        self._run_button.setMinimumHeight(40)
        self._run_button.setStyleSheet("QPushButton { font-weight: bold; font-size: 14px; }")
        layout.addWidget(self._run_button)

        # Coverage heatmap group
        coverage_group = QGroupBox("Coverage Matrix")
        coverage_layout = QVBoxLayout(coverage_group)

        self._heatmap = CoverageHeatmapWidget()
        coverage_layout.addWidget(self._heatmap)

        layout.addWidget(coverage_group)

        # Error metrics group
        self._metrics_group = QGroupBox("Error Metrics")
        metrics_layout = QVBoxLayout(self._metrics_group)

        self._rmse_label = QLabel("RMSE: --")
        metrics_layout.addWidget(self._rmse_label)

        # Per-camera error labels (will be populated dynamically)
        self._camera_error_labels: list[QLabel] = []

        layout.addWidget(self._metrics_group)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._status_label)

        # Stretch at bottom to push controls up
        layout.addStretch()

        return sidebar

    def _create_main_area(self) -> QWidget:
        """Create the main visualization area."""
        main_area = QWidget()
        layout = QVBoxLayout(main_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Storyboard (4-panel 3D view)
        self._storyboard = StoryboardView()
        layout.addWidget(self._storyboard, stretch=1)

        # Frame slider at bottom
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(8, 4, 8, 4)

        slider_label = QLabel("Frame:")
        slider_layout.addWidget(slider_label)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)  # Updated when scene loads
        self._frame_slider.setEnabled(False)
        slider_layout.addWidget(self._frame_slider, stretch=1)

        self._frame_display = QLabel("0 / 0")
        self._frame_display.setMinimumWidth(60)
        slider_layout.addWidget(self._frame_display)

        layout.addLayout(slider_layout)

        return main_area

    def _connect_signals(self) -> None:
        """Wire up signal connections between UI and presenter."""
        # UI -> Presenter
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self._run_button.clicked.connect(self._presenter.run_pipeline)
        self._frame_slider.valueChanged.connect(self._presenter.set_frame)

        # Presenter -> UI
        self._presenter.scene_changed.connect(self._on_scene_changed)
        self._presenter.filter_changed.connect(self._on_filter_changed)
        self._presenter.pipeline_started.connect(self._on_pipeline_started)
        self._presenter.pipeline_finished.connect(self._on_pipeline_finished)
        self._presenter.pipeline_failed.connect(self._on_pipeline_failed)
        self._presenter.frame_changed.connect(self._on_frame_changed)

    def _on_preset_changed(self, index: int) -> None:
        """Handle preset selection change."""
        preset_names = list(PRESETS.keys())
        if 0 <= index < len(preset_names):
            config = PRESETS[preset_names[index]]
            self._presenter.set_config(config)
            self._status_label.setText("Ready")
            self._status_label.setStyleSheet("color: #888; font-style: italic;")

    def _on_scene_changed(self, scene) -> None:
        """Handle scene rebuild from presenter."""
        self._storyboard.set_scene(scene)

        # Update frame slider range
        n_frames = scene.n_frames
        self._frame_slider.blockSignals(True)
        self._frame_slider.setMaximum(max(0, n_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.setEnabled(n_frames > 1)
        self._frame_slider.blockSignals(False)

        self._update_frame_display(0, n_frames)

    def _on_filter_changed(self, coverage) -> None:
        """Handle filter/coverage matrix update from presenter."""
        killed_set = set(self._presenter.filter_config.killed_linkages)
        self._heatmap.set_data(coverage, killed_set)

    def _on_pipeline_started(self) -> None:
        """Handle pipeline start."""
        self._run_button.setEnabled(False)
        self._run_button.setText("Running...")
        self._status_label.setText("Running pipeline...")
        self._status_label.setStyleSheet("color: #5599ff; font-style: italic;")

    def _on_pipeline_finished(self, result: PipelineResult) -> None:
        """Handle pipeline completion."""
        self._run_button.setEnabled(True)
        self._run_button.setText("Run Pipeline")

        # Update storyboard with results
        self._storyboard.set_result(result)

        # Update error metrics display
        self._update_metrics_display(result)

        # Update status based on result
        errors = []
        if result.bootstrap_error:
            errors.append(f"Bootstrap: {result.bootstrap_error}")
        if result.optimization_error:
            errors.append(f"Optimize: {result.optimization_error}")
        if result.alignment_error:
            errors.append(f"Align: {result.alignment_error}")

        if errors:
            self._status_label.setText("Complete (with errors)")
            self._status_label.setStyleSheet("color: #ffaa00; font-style: italic;")
            self._status_label.setToolTip(chr(92) + "n".join(errors))
        else:
            self._status_label.setText("Complete")
            self._status_label.setStyleSheet("color: #55ff55; font-style: italic;")
            self._status_label.setToolTip("")

    def _on_pipeline_failed(self, error: str) -> None:
        """Handle pipeline failure."""
        self._run_button.setEnabled(True)
        self._run_button.setText("Run Pipeline")
        self._status_label.setText("Failed")
        self._status_label.setStyleSheet("color: #ff5555; font-style: italic;")
        self._status_label.setToolTip(error)
        logger.error(f"Pipeline failed: {error}")

    def _on_frame_changed(self, frame: int) -> None:
        """Handle frame navigation from presenter."""
        # Update slider without triggering signal back to presenter
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(frame)
        self._frame_slider.blockSignals(False)

        # Update storyboard
        self._storyboard.set_frame(frame)

        # Update display
        self._update_frame_display(frame, self._presenter.n_frames)

    def _update_frame_display(self, frame: int, total: int) -> None:
        """Update the frame number display."""
        self._frame_display.setText(f"{frame + 1} / {total}")

    def _update_metrics_display(self, result: PipelineResult) -> None:
        """Update error metrics display with results from pipeline."""
        metrics_layout = self._metrics_group.layout()

        # Update reprojection RMSE
        if result.reprojection_rmse is not None:
            self._rmse_label.setText(f"RMSE: {result.reprojection_rmse:.3f} px")
        else:
            self._rmse_label.setText("RMSE: --")

        # Clear existing camera error labels
        for label in self._camera_error_labels:
            metrics_layout.removeWidget(label)
            label.deleteLater()
        self._camera_error_labels.clear()

        # Add camera error labels
        if result.camera_metrics:
            for metrics in result.camera_metrics:
                label_text = (
                    f"C{metrics.port}: {metrics.rotation_error_deg:.2f}° / {metrics.translation_error_mm:.1f}mm"
                )
                label = QLabel(label_text)
                label.setStyleSheet("font-family: monospace; font-size: 11px;")
                metrics_layout.addWidget(label)
                self._camera_error_labels.append(label)

    # --- Lifecycle Management ---

    def cleanup(self) -> None:
        """Explicit cleanup - MUST be called before destruction.

        Note: closeEvent is NOT reliable for tab widgets because
        removeTab() + deleteLater() doesn't trigger closeEvent.
        """
        self._storyboard.cleanup()
        self._presenter.cancel_pipeline()
        logger.info("ExplorerTab cleaned up")

    def suspend_vtk(self) -> None:
        """Pause VTK rendering when tab is not visible."""
        self._storyboard.suspend_vtk()

    def resume_vtk(self) -> None:
        """Resume VTK rendering when tab becomes visible."""
        self._storyboard.resume_vtk()

    def closeEvent(self, event) -> None:
        """Defensive cleanup if explicit cleanup wasn't called."""
        self.cleanup()
        super().closeEvent(event)
