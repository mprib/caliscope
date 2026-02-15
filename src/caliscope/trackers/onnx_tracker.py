"""ONNX-based pose tracker with generic model card system.

Supports two model output formats:
- SimCC (Simulated Coordinate Classification): 1D probability vectors
- Heatmap: 2D spatial probability maps

Model configuration is loaded from a TOML "model card" file.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore[reportMissingImports]  # optional dependency

    _ONNXRUNTIME_AVAILABLE = True
except ImportError:
    _ONNXRUNTIME_AVAILABLE = False

from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points
from caliscope.trackers.model_card import ModelCard
from caliscope.trackers.model_decode import decode_heatmap, decode_simcc

logger = logging.getLogger(__name__)


class OnnxTracker(Tracker):
    """Generic ONNX pose tracker configured via ModelCard.

    Loads any pose estimation model that outputs either SimCC vectors or heatmaps.
    First tracker to populate PointPacket.confidence field.
    """

    def __init__(self, card: ModelCard) -> None:
        """Create inference session from model card.

        Args:
            card: Model configuration and metadata

        Raises:
            ImportError: If onnxruntime not installed
            FileNotFoundError: If ONNX model file doesn't exist
        """
        if not _ONNXRUNTIME_AVAILABLE:
            raise ImportError("onnxruntime is required for OnnxTracker. Install with: pip install caliscope[onnx]")

        self.card = card

        # Check ONNX file exists
        if not card.onnx_exists:
            raise FileNotFoundError(f"ONNX model not found: {card.model_path}")

        # Create onnxruntime session (CPU only)
        logger.info(f"Loading ONNX model: {card.model_path}")
        # ort is guaranteed bound here: _ONNXRUNTIME_AVAILABLE guard above raises on False
        self.session = ort.InferenceSession(  # type: ignore[reportPossiblyUnboundVariable]
            str(card.model_path),
            providers=["CPUExecutionProvider"],
        )

        # Get input name from model
        self.input_name = self.session.get_inputs()[0].name

        logger.info(
            f"OnnxTracker initialized: {card.name}, "
            f"format={card.format}, input_size={card.input_width}x{card.input_height}"
        )

    @property
    def name(self) -> str:
        """Return tracker name derived from ONNX filename stem."""
        return f"ONNX_{self.card.model_path.stem}"

    def _preprocess_simcc(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for SimCC format models (RTMPose).

        Steps:
        1. Resize to model input size
        2. BGR -> RGB
        3. Normalize with ImageNet mean/std
        4. Transpose to (1, 3, H, W) float32

        Args:
            frame: BGR image from cv2.VideoCapture

        Returns:
            Preprocessed array ready for inference
        """
        # Resize to model input size (cv2.resize expects (width, height))
        resized = cv2.resize(frame, (self.card.input_width, self.card.input_height), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Convert to float and normalize
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std

        # Transpose (H, W, C) -> (C, H, W) and add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))  # (3, H, W)
        batched = np.expand_dims(transposed, axis=0)  # (1, 3, H, W)

        return batched

    def _preprocess_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for heatmap format models (SLEAP).

        Steps:
        1. Resize to model input size
        2. BGR -> grayscale
        3. Shape to (1, 1, H, W) uint8

        Args:
            frame: BGR image from cv2.VideoCapture

        Returns:
            Preprocessed array ready for inference
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.card.input_width, self.card.input_height), interpolation=cv2.INTER_LINEAR)

        # BGR -> grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Shape to (1, 1, H, W) uint8
        batched = np.expand_dims(np.expand_dims(gray, axis=0), axis=0)

        return batched

    def get_points(self, frame: np.ndarray, port: int = 0, rotation_count: int = 0) -> PointPacket:
        """Detect pose keypoints in frame.

        Process:
        1. Apply rotation if rotation_count != 0
        2. Store original dimensions for coordinate scaling
        3. Resize to model input size
        4. Preprocess based on format (simcc or heatmap)
        5. Run inference
        6. Decode based on format
        7. Scale coordinates back to original frame size
        8. Unrotate points if rotation_count != 0
        9. Apply confidence threshold
        10. Return PointPacket with point_id, img_loc, confidence

        Args:
            frame: BGR image from cv2.VideoCapture
            port: Camera identifier (unused by tracker, passed through)
            rotation_count: Number of 90-degree rotations to apply

        Returns:
            PointPacket with detected keypoints and confidence scores
        """
        # Apply rotation if needed
        if rotation_count != 0:
            frame = apply_rotation(frame, rotation_count)

        # Store original dimensions for coordinate scaling
        original_height, original_width = frame.shape[:2]

        # Preprocess based on format
        if self.card.format == "simcc":
            preprocessed = self._preprocess_simcc(frame)
        else:  # heatmap
            preprocessed = self._preprocess_heatmap(frame)

        # Run inference
        outputs = self.session.run(None, {self.input_name: preprocessed})

        # Decode based on format
        if self.card.format == "simcc":
            # SimCC outputs are two tensors: simcc_x and simcc_y
            if len(outputs) != 2:
                raise ValueError(f"SimCC format expects 2 outputs, got {len(outputs)}")
            keypoints, confidence = decode_simcc(outputs[0], outputs[1])
        else:  # heatmap
            # Heatmap output is a single tensor (batch, K, H, W)
            # Remove batch dimension before passing to decode_heatmap
            if len(outputs) != 1:
                raise ValueError(f"Heatmap format expects 1 output, got {len(outputs)}")
            heatmaps = outputs[0][0]  # Remove batch dimension: (K, H, W)
            keypoints, confidence = decode_heatmap(heatmaps)

        # Scale coordinates back to original frame size
        if self.card.format == "simcc":
            # SimCC keypoints are in model input space
            scale_x = float(original_width) / float(self.card.input_width)
            scale_y = float(original_height) / float(self.card.input_height)
        else:  # heatmap
            # Heatmap keypoints are in heatmap space (which matches model input size)
            # Get heatmap dimensions from the output
            heatmap_height, heatmap_width = outputs[0].shape[2:]
            scale_x = float(original_width) / float(heatmap_width)
            scale_y = float(original_height) / float(heatmap_height)

        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # Unrotate points if needed
        if rotation_count != 0:
            keypoints = unrotate_points(keypoints, rotation_count, original_width, original_height)

        # Apply confidence threshold and filter
        mask = confidence >= self.card.confidence_threshold
        filtered_keypoints = keypoints[mask]
        filtered_confidence = confidence[mask]

        # Generate point IDs (indices of keypoints that passed threshold)
        point_ids = np.arange(len(confidence), dtype=np.int32)[mask]

        return PointPacket(
            point_id=point_ids,
            img_loc=filtered_keypoints,
            obj_loc=None,
            confidence=filtered_confidence,
        )

    def get_point_name(self, point_id: int) -> str:
        """Map point ID to landmark name.

        Args:
            point_id: Keypoint index

        Returns:
            Landmark name from model card [points] section
        """
        return self.card.point_id_to_name.get(point_id, str(point_id))

    def scatter_draw_instructions(self, point_id: int) -> dict:
        """Return drawing parameters for visualizing keypoints.

        Returns cyan circles to distinguish from other trackers:
        - ArUco: green
        - Charuco: blue
        - ONNX: cyan

        Args:
            point_id: Keypoint index (unused, all points drawn the same)

        Returns:
            Dictionary with radius, color, thickness for cv2.circle
        """
        return {
            "radius": 5,
            "color": (255, 255, 0),  # Cyan in BGR
            "thickness": -1,  # Filled circle
        }

    @property
    def wireframe_toml_path(self) -> Path | None:
        """Return path to wireframe definition.

        Wireframe is now embedded directly in ModelCard.wireframe,
        not stored in a separate TOML file. This property exists
        for backward compatibility with other trackers.

        Returns:
            None (wireframe accessible via card.wireframe)
        """
        return None

    def cleanup(self) -> None:
        """Release onnxruntime session resources."""
        del self.session
        logger.debug(f"OnnxTracker cleaned up: {self.card.name}")
