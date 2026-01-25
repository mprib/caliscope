"""Presenter for multi-camera synchronized video processing.

Coordinates batch processing of synchronized video to extract 2D landmarks.
Wraps process_synchronized_recording() with Qt signals for UI integration.

This is a "scratchpad" presenter - processing results are transient until
emitted to the Coordinator for persistence.
"""

import logging
from enum import Enum, auto
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from caliscope.cameras.camera_array import CameraData
from caliscope.task_manager.task_handle import TaskHandle
from caliscope.task_manager.task_manager import TaskManager
from caliscope.task_manager.task_state import TaskState
from caliscope.tracker import Tracker

logger = logging.getLogger(__name__)


class MultiCameraProcessingState(Enum):
    """Workflow states for multi-camera processing.

    States are computed from internal reality, not stored separately.
    This prevents state/reality divergence.
    """

    UNCONFIGURED = auto()  # Missing required inputs
    READY = auto()  # Can start processing
    PROCESSING = auto()  # Background task running
    COMPLETE = auto()  # Results available


class MultiCameraProcessingPresenter(QObject):
    """Presenter for multi-camera synchronized video processing.

    Manages the extraction of 2D landmarks from synchronized multi-camera
    video recordings. Provides rotation control per camera and throttled
    thumbnail updates during processing.

    Signals:
        state_changed: Emitted when computed state changes. View updates UI.

    Usage:
        presenter = MultiCameraProcessingPresenter(task_manager, tracker)
        presenter.set_recording_dir(path)
        presenter.set_cameras(cameras)
        # Additional methods (start_processing, etc.) to be added in Task 2.2
    """

    # State signals
    state_changed = Signal(MultiCameraProcessingState)

    def __init__(
        self,
        task_manager: TaskManager,
        tracker: Tracker,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the presenter.

        Args:
            task_manager: TaskManager for background processing
            tracker: Tracker for 2D point extraction
            parent: Optional Qt parent
        """
        super().__init__(parent)

        self._task_manager = task_manager
        self._tracker = tracker

        # Configuration state (set externally)
        self._recording_dir: Path | None = None
        self._cameras: dict[int, CameraData] = {}

        # Processing state (managed internally)
        self._task_handle: TaskHandle | None = None
        self._result = None  # Will be ImagePoints when implemented

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> MultiCameraProcessingState:
        """Compute current state from internal reality - never stale."""
        if self._result is not None:
            return MultiCameraProcessingState.COMPLETE

        if self._task_handle is not None and self._task_handle.state == TaskState.RUNNING:
            return MultiCameraProcessingState.PROCESSING

        if self._recording_dir is None or not self._cameras:
            return MultiCameraProcessingState.UNCONFIGURED

        return MultiCameraProcessingState.READY

    @property
    def recording_dir(self) -> Path | None:
        """Current recording directory."""
        return self._recording_dir

    @property
    def cameras(self) -> dict[int, CameraData]:
        """Current camera configuration (copy)."""
        return dict(self._cameras)

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def set_recording_dir(self, path: Path) -> None:
        """Set the recording directory.

        Resets any existing results.

        Args:
            path: Directory containing port_N.mp4 and frame_timestamps.csv
        """
        if self.state == MultiCameraProcessingState.PROCESSING:
            logger.warning("Cannot change recording_dir while processing")
            return

        self._recording_dir = path
        self._reset_results()
        self._emit_state_changed()

    def set_cameras(self, cameras: dict[int, CameraData]) -> None:
        """Set the camera configuration.

        Makes a shallow copy of the cameras dict. Resets any existing results.

        Args:
            cameras: Camera data by port
        """
        if self.state == MultiCameraProcessingState.PROCESSING:
            logger.warning("Cannot change cameras while processing")
            return

        # Shallow copy - we'll modify rotation_count on these objects in Task 2.3
        self._cameras = {port: cam for port, cam in cameras.items()}
        self._reset_results()
        self._emit_state_changed()

    # -------------------------------------------------------------------------
    # Private: Helpers
    # -------------------------------------------------------------------------

    def _reset_results(self) -> None:
        """Clear processing results (not configuration)."""
        self._result = None
        self._task_handle = None

    def _emit_state_changed(self) -> None:
        """Emit state_changed signal with current computed state."""
        current_state = self.state
        logger.debug(f"State changed to {current_state}")
        self.state_changed.emit(current_state)
