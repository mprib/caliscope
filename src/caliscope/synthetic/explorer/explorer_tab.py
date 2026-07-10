"""Synthetic Calibration Explorer — interactive testbed for the calibration pipeline.

PURPOSE
  Run the full calibration pipeline (bootstrap → optimize → align) on synthetic
  scenes with known ground truth, and see exactly how well the pipeline recovers
  camera poses. Every number in the UI is checkable against ground truth because
  the scene generated the data.

WHAT YOU CAN EXPLORE
  - Scene geometry: how camera placement, board shape, and trajectory affect
    calibration accuracy (Default Ring vs Sparse Coverage vs Chain-Linked).
  - Lens profiles: WEBCAM vs MACHINE_VISION vs IDEAL, and how distortion
    coefficients propagate through the pipeline.
  - Intrinsic sensitivity: the "Perturbed Intrinsics" presets feed wrong focal
    lengths into the pipeline and show the resulting extrinsic degradation.
    Worse intrinsics → worse extrinsics, visible in the error metrics.
  - Visibility and cheirality: single-sided boards, backward cameras, and
    how the projection model handles edge cases.

LAYOUT
  Left sidebar: preset selector, Run Pipeline button, camera intrinsics table,
  coverage heatmap, per-camera error metrics (rotation/translation/RMSE).

  Main area: 4-panel storyboard (ground truth, bootstrapped, optimized, aligned)
  with a frame slider for scrubbing through the trajectory.

INTRINSICS TABLE
  Shows per-camera focal length and Brown-Conrady distortion coefficients.
  For perturbed presets, columns split into "truth" (what generated the data)
  and "input" (what the pipeline sees). Perturbed values render in red.

ADDING A PRESET
  1. Write a factory in scene_factories.py returning SyntheticScene
  2. Add a ScenePreset entry to SCENE_PRESETS below
  3. For intrinsic perturbation experiments, set the perturbation field
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from caliscope.cameras.camera_array import CameraArray
from caliscope.synthetic.camera_synthesizer import IntrinsicPerturbation
from caliscope.synthetic.explorer.presenter import ExplorerPresenter, PipelineResult
from caliscope.synthetic.explorer.widgets import CoverageHeatmapWidget, StoryboardView
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.scene_factories import (
    aruco_multi_object_scene,
    charuco_target_scene,
    cheirality_demo_scene,
    default_ring_scene,
    machine_vision_scene,
    quick_test_scene,
    sparse_coverage_scene,
    visibility_culling_scene,
    wand_scene,
)
from caliscope.task_manager.task_manager import TaskManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenePreset:
    factory: Callable[[], SyntheticScene]
    perturbation: IntrinsicPerturbation | None = None


def _preset(factory: Callable[[], SyntheticScene]) -> ScenePreset:
    return ScenePreset(factory=factory)


SCENE_PRESETS: dict[str, ScenePreset] = {
    "Default Ring (4 cameras, full orbit)": _preset(default_ring_scene),
    "Sparse Coverage (180° arc)": _preset(sparse_coverage_scene),
    "Quick Test (5 frames)": _preset(quick_test_scene),
    "Cheirality Demo (forward vs backward camera)": _preset(cheirality_demo_scene),
    "Visibility Culling (single-sided board)": _preset(visibility_culling_scene),
    "ArUco Multi-Object (mobile + static marker)": _preset(aruco_multi_object_scene),
    "Wand Scene (2 linked + 2 static ArUcos)": _preset(wand_scene),
    "Charuco Target (double-sided board)": _preset(charuco_target_scene),
    "Machine Vision Lens (KITTI-class barrel)": _preset(machine_vision_scene),
    "Perturbed Intrinsics (3% focal error)": ScenePreset(
        factory=default_ring_scene,
        perturbation=IntrinsicPerturbation(f_scale=1.03),
    ),
    "Perturbed Intrinsics (10% focal error)": ScenePreset(
        factory=default_ring_scene,
        perturbation=IntrinsicPerturbation(f_scale=1.10),
    ),
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

        # Left sidebar (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar = self._create_sidebar()
        scroll.setWidget(sidebar)
        splitter.addWidget(scroll)

        # Main visualization area
        main_area = self._create_main_area()
        splitter.addWidget(main_area)

        # Set initial splitter sizes (sidebar ~350px, main area gets the rest)
        splitter.setSizes([350, 850])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

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
        self._preset_combo.addItems(list(SCENE_PRESETS.keys()))
        preset_layout.addWidget(self._preset_combo)

        layout.addWidget(preset_group)

        # Run button
        self._run_button = QPushButton("Run Pipeline")
        self._run_button.setMinimumHeight(40)
        self._run_button.setStyleSheet("QPushButton { font-weight: bold; font-size: 14px; }")
        layout.addWidget(self._run_button)

        # Camera intrinsics group
        intrinsics_group = QGroupBox("Camera Intrinsics")
        intrinsics_layout = QVBoxLayout(intrinsics_group)

        self._intrinsics_table = QTableWidget()
        self._intrinsics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._intrinsics_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._intrinsics_table.setStyleSheet("font-family: monospace; font-size: 12px;")
        self._intrinsics_table.verticalHeader().setDefaultSectionSize(22)
        intrinsics_layout.addWidget(self._intrinsics_table)

        layout.addWidget(intrinsics_group)

        # Coverage heatmap group
        coverage_group = QGroupBox("Coverage Matrix")
        coverage_layout = QVBoxLayout(coverage_group)

        self._heatmap = CoverageHeatmapWidget()
        coverage_layout.addWidget(self._heatmap)

        layout.addWidget(coverage_group)

        # Error metrics group
        self._metrics_group = QGroupBox("Error Metrics")
        self._metrics_layout = QVBoxLayout(self._metrics_group)

        self._rmse_label = QLabel("RMSE: --")
        self._metrics_layout.addWidget(self._rmse_label)

        self._camera_error_labels: list[QLabel] = []

        layout.addWidget(self._metrics_group)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._status_label)

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
        self._frame_slider.setMaximum(0)
        self._frame_slider.setEnabled(False)
        slider_layout.addWidget(self._frame_slider, stretch=1)

        self._frame_display = QLabel("0 / 0")
        self._frame_display.setMinimumWidth(60)
        slider_layout.addWidget(self._frame_display)

        layout.addLayout(slider_layout)

        return main_area

    def _connect_signals(self) -> None:
        """Wire up signal connections between UI and presenter."""
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self._run_button.clicked.connect(self._presenter.run_pipeline)
        self._frame_slider.valueChanged.connect(self._presenter.set_frame)

        self._presenter.scene_changed.connect(self._on_scene_changed)
        self._presenter.filter_changed.connect(self._on_filter_changed)
        self._presenter.pipeline_started.connect(self._on_pipeline_started)
        self._presenter.pipeline_finished.connect(self._on_pipeline_finished)
        self._presenter.pipeline_failed.connect(self._on_pipeline_failed)
        self._presenter.frame_changed.connect(self._on_frame_changed)

    def _on_preset_changed(self, index: int) -> None:
        """Handle preset selection change."""
        preset_names = list(SCENE_PRESETS.keys())
        if 0 <= index < len(preset_names):
            preset = SCENE_PRESETS[preset_names[index]]
            self._presenter.set_scene(preset.factory(), preset.perturbation)
            self._status_label.setText("Ready")
            self._status_label.setStyleSheet("color: #888; font-style: italic;")

    def _on_scene_changed(self, scene: SyntheticScene) -> None:
        """Handle scene rebuild from presenter."""
        self._storyboard.set_scene(scene)

        n_frames = scene.n_frames
        self._frame_slider.blockSignals(True)
        self._frame_slider.setMaximum(max(0, n_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.setEnabled(n_frames > 1)
        self._frame_slider.blockSignals(False)

        self._update_frame_display(0, n_frames)
        self._update_intrinsics_table(scene.camera_array, self._presenter.perturbation)
        self._reset_metrics()

    def _update_intrinsics_table(
        self,
        camera_array: CameraArray,
        perturbation: IntrinsicPerturbation | None,
    ) -> None:
        """Populate the intrinsics table from the camera array."""
        cam_ids = sorted(camera_array.cameras.keys())
        n_cams = len(cam_ids)

        columns = ["cam", "f", "k1", "k2", "p1", "p2", "k3"]
        if perturbation is not None:
            columns = ["cam", "f (truth)", "f (input)", "k1 (truth)", "k1 (input)", "k2", "p1", "p2", "k3"]

        table = self._intrinsics_table
        table.setRowCount(n_cams)
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)

        for row, cam_id in enumerate(cam_ids):
            cam = camera_array.cameras[cam_id]
            matrix = cam.matrix
            dist = cam.distortions

            if matrix is None or dist is None:
                continue

            f_truth = matrix[0, 0]
            k1_truth = dist[0]

            col = 0
            table.setItem(row, col, QTableWidgetItem(str(cam_id)))
            col += 1

            if perturbation is not None:
                f_input = f_truth * perturbation.f_scale
                k1_input = k1_truth + perturbation.k1_delta

                table.setItem(row, col, QTableWidgetItem(f"{f_truth:.1f}"))
                col += 1
                item = QTableWidgetItem(f"{f_input:.1f}")
                if abs(perturbation.f_scale - 1.0) > 1e-6:
                    item.setForeground(Qt.GlobalColor.red)
                table.setItem(row, col, item)
                col += 1
                table.setItem(row, col, QTableWidgetItem(f"{k1_truth:.4f}"))
                col += 1
                item = QTableWidgetItem(f"{k1_input:.4f}")
                if abs(perturbation.k1_delta) > 1e-6:
                    item.setForeground(Qt.GlobalColor.red)
                table.setItem(row, col, item)
                col += 1
            else:
                table.setItem(row, col, QTableWidgetItem(f"{f_truth:.1f}"))
                col += 1
                table.setItem(row, col, QTableWidgetItem(f"{k1_truth:.4f}"))
                col += 1

            table.setItem(row, col, QTableWidgetItem(f"{dist[1]:.4f}"))
            col += 1
            table.setItem(row, col, QTableWidgetItem(f"{dist[2]:.4f}"))
            col += 1
            table.setItem(row, col, QTableWidgetItem(f"{dist[3]:.4f}"))
            col += 1
            table.setItem(row, col, QTableWidgetItem(f"{dist[4]:.4f}"))

        header = table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setMaximumHeight(22 * n_cams + table.horizontalHeader().height() + 4)

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

        self._storyboard.set_result(result)
        self._update_metrics_display(result)

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
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(frame)
        self._frame_slider.blockSignals(False)

        self._storyboard.set_frame(frame)
        self._update_frame_display(frame, self._presenter.n_frames)

    def _update_frame_display(self, frame: int, total: int) -> None:
        """Update the frame number display."""
        self._frame_display.setText(f"{frame + 1} / {total}")

    def _reset_metrics(self) -> None:
        """Clear error metrics when a new scenario is selected."""
        self._rmse_label.setText("RMSE: --")
        for label in self._camera_error_labels:
            self._metrics_layout.removeWidget(label)
            label.deleteLater()
        self._camera_error_labels.clear()

    def _update_metrics_display(self, result: PipelineResult) -> None:
        """Update error metrics display with results from pipeline."""
        metrics_layout = self._metrics_layout

        if result.reprojection_rmse is not None:
            self._rmse_label.setText(f"RMSE: {result.reprojection_rmse:.3f} px")
        else:
            self._rmse_label.setText("RMSE: --")

        for label in self._camera_error_labels:
            metrics_layout.removeWidget(label)
            label.deleteLater()
        self._camera_error_labels.clear()

        if result.camera_metrics:
            for metrics in result.camera_metrics:
                label_text = (
                    f"C{metrics.cam_id}: {metrics.rotation_error_deg:.2f}° / "
                    f"{metrics.translation_error_m * 1000:.1f}mm | {metrics.reprojection_rmse:.2f}px"
                )
                label = QLabel(label_text)
                label.setStyleSheet("font-family: monospace; font-size: 15px;")
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

    def suspend_rendering(self) -> None:
        """Pause 3D rendering when tab is not visible."""
        self._storyboard.suspend_rendering()

    def resume_rendering(self) -> None:
        """Resume 3D rendering when tab becomes visible."""
        self._storyboard.resume_rendering()

    def closeEvent(self, event) -> None:
        """Defensive cleanup if explicit cleanup wasn't called."""
        self.cleanup()
        super().closeEvent(event)
