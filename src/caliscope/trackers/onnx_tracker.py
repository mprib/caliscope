"""ONNX-based pose tracker with generic model card system.

Supports two model output formats:
- SimCC (Simulated Coordinate Classification): 1D probability vectors
- Heatmap: 2D spatial probability maps

Model configuration is loaded from a TOML "model card" file.
"""

import logging

import cv2
import numpy as np
import onnxruntime as ort  # type: ignore[reportMissingImports]  # no type stubs

from caliscope.packets import PointPacket
from caliscope.tracker import Tracker, WireFrameView
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
            FileNotFoundError: If ONNX model file doesn't exist
        """
        self.card = card

        # Check ONNX file exists
        if not card.onnx_exists:
            raise FileNotFoundError(f"ONNX model not found: {card.model_path}")

        # Create onnxruntime session (CPU only)
        logger.info(f"Loading ONNX model: {card.model_path}")
        self.session = ort.InferenceSession(
            str(card.model_path),
            providers=["CPUExecutionProvider"],
        )

        # Get input name from model
        self.input_name = self.session.get_inputs()[0].name

        # Tracking state: per-camera previous bounding box in post-rotation coords.
        # Keyed by cam_id to isolate state across cameras (single tracker instance
        # is shared across all cameras — see Reconstructor/SynchronizedStreamManager).
        self._prev_bboxes: dict[int, tuple[int, int, int, int]] = {}

        logger.info(
            f"OnnxTracker initialized: {card.name}, "
            f"format={card.format}, input_size={card.input_width}x{card.input_height}"
        )

    @property
    def name(self) -> str:
        """Return tracker name derived from ONNX filename stem."""
        return f"ONNX_{self.card.model_path.stem}"

    def _preprocess_simcc(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Preprocess frame for SimCC format models (RTMPose).

        Letterboxes the frame to preserve aspect ratio, then normalizes.
        Returns the preprocessed tensor plus transform parameters needed
        to map model-space coordinates back to original frame coordinates.

        Args:
            frame: BGR image from cv2.VideoCapture

        Returns:
            Tuple of (preprocessed, scale, pad_x, pad_y) where:
            - preprocessed: (1, 3, H, W) float32 tensor
            - scale: resize scale factor applied to the original frame
            - pad_x: horizontal padding in model input pixels
            - pad_y: vertical padding in model input pixels
        """
        src_h, src_w = frame.shape[:2]
        dst_w, dst_h = self.card.input_width, self.card.input_height

        # Compute uniform scale that fits the frame inside the model input
        scale = min(dst_w / src_w, dst_h / src_h)
        scaled_w = int(src_w * scale)
        scaled_h = int(src_h * scale)

        # Padding to center the scaled image in the model input
        pad_x = (dst_w - scaled_w) // 2
        pad_y = (dst_h - scaled_h) // 2

        # Resize preserving aspect ratio, then place on black canvas
        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w] = resized

        # BGR -> RGB
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std

        # Transpose (H, W, C) -> (C, H, W) and add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))  # (3, H, W)
        batched = np.expand_dims(transposed, axis=0)  # (1, 3, H, W)

        return batched, scale, pad_x, pad_y

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

    def _infer_simcc(self, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run SimCC inference on a BGR image region.

        Returns keypoints in the region's coordinate space (not original frame).
        """
        preprocessed, lb_scale, pad_x, pad_y = self._preprocess_simcc(region)
        outputs = self.session.run(None, {self.input_name: preprocessed})

        if len(outputs) != 2:
            raise ValueError(f"SimCC format expects 2 outputs, got {len(outputs)}")
        simcc_x = np.asarray(outputs[0])
        simcc_y = np.asarray(outputs[1])
        keypoints, confidence = decode_simcc(simcc_x, simcc_y)

        # Undo letterbox: subtract padding offset, then invert the scale
        keypoints[:, 0] = (keypoints[:, 0] - pad_x) / lb_scale
        keypoints[:, 1] = (keypoints[:, 1] - pad_y) / lb_scale

        return keypoints, confidence

    def _infer_heatmap(
        self, region: np.ndarray, original_width: int, original_height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run heatmap inference on a BGR image region."""
        preprocessed = self._preprocess_heatmap(region)
        outputs = self.session.run(None, {self.input_name: preprocessed})

        if len(outputs) != 1:
            raise ValueError(f"Heatmap format expects 1 output, got {len(outputs)}")
        raw_heatmaps = np.asarray(outputs[0])
        heatmaps = raw_heatmaps[0]
        keypoints, confidence = decode_heatmap(heatmaps)

        heatmap_height, heatmap_width = raw_heatmaps.shape[2:]
        scale_x = float(original_width) / float(heatmap_width)
        scale_y = float(original_height) / float(heatmap_height)
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return keypoints, confidence

    def _bbox_from_keypoints(
        self, keypoints: np.ndarray, confidence: np.ndarray, frame_w: int, frame_h: int, padding: float = 0.3
    ) -> tuple[int, int, int, int]:
        """Compute a padded, aspect-ratio-correct bounding box from keypoints.

        Takes detected keypoints, adds padding, then expands to match the
        model's aspect ratio (3:4 portrait for RTMPose). Clamps to frame bounds.
        """
        mask = confidence >= self.card.confidence_threshold
        if not np.any(mask):
            return (0, 0, frame_w, frame_h)

        valid_kps = keypoints[mask]
        x_min, y_min = valid_kps.min(axis=0)
        x_max, y_max = valid_kps.max(axis=0)

        # Add padding proportional to bbox size
        w = x_max - x_min
        h = y_max - y_min
        pad_w = w * padding
        pad_h = h * padding
        x_min -= pad_w
        y_min -= pad_h
        x_max += pad_w
        y_max += pad_h

        # Expand to match model aspect ratio (width/height)
        target_aspect = self.card.input_width / self.card.input_height
        box_w = x_max - x_min
        box_h = y_max - y_min
        current_aspect = box_w / max(box_h, 1)

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        if current_aspect > target_aspect:
            # Too wide — expand height
            box_h = box_w / target_aspect
        else:
            # Too tall — expand width
            box_w = box_h * target_aspect

        x1 = int(max(0, cx - box_w / 2))
        y1 = int(max(0, cy - box_h / 2))
        x2 = int(min(frame_w, cx + box_w / 2))
        y2 = int(min(frame_h, cy + box_h / 2))

        # Ensure bbox has at least 1px in each dimension (single-keypoint edge case)
        if x2 <= x1:
            x2 = min(x1 + 1, frame_w)
        if y2 <= y1:
            y2 = min(y1 + 1, frame_h)

        return (x1, y1, x2, y2)

    def _scan_positions(self, frame_w: int, frame_h: int) -> list[tuple[int, int, int, int]]:
        """Generate sliding window crop positions across the frame.

        Creates aspect-ratio-correct windows at full frame height,
        sliding horizontally with 50% overlap.
        """
        target_aspect = self.card.input_width / self.card.input_height
        crop_h = frame_h
        crop_w = int(crop_h * target_aspect)

        if crop_w >= frame_w:
            # Frame is narrower than one crop — single centered crop
            return [(0, 0, frame_w, frame_h)]

        # Slide with 50% overlap
        step = max(crop_w // 2, 1)
        positions = []
        x = 0
        while x + crop_w <= frame_w:
            positions.append((x, 0, x + crop_w, frame_h))
            x += step
        # Ensure we cover the right edge
        if positions[-1][2] < frame_w:
            positions.append((frame_w - crop_w, 0, frame_w, frame_h))

        return positions

    def _detect_in_region(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a crop region, return keypoints in original frame coords."""
        if x2 <= x1 or y2 <= y1:
            n = len(self.card.point_name_to_id)
            return np.zeros((n, 2), dtype=np.float32), np.zeros(n, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]

        if self.card.format == "simcc":
            keypoints, confidence = self._infer_simcc(crop)
        else:
            keypoints, confidence = self._infer_heatmap(crop, x2 - x1, y2 - y1)

        # Offset from crop coords to original frame coords
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1

        return keypoints, confidence

    def get_points(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        """Detect pose keypoints in frame using three-tier search.

        Tier 1: Crop to previous detection (fast, common case)
        Tier 2: Full-frame letterbox (cold start or lost tracking)
        Tier 3: Sliding window scan (thorough search when full-frame fails)

        After any successful detection, stores the bounding box for Tier 1
        on the next frame.

        Args:
            frame: BGR image from cv2.VideoCapture
            cam_id: Camera identifier (unused by tracker, passed through)
            rotation_count: Number of 90-degree rotations to apply

        Returns:
            PointPacket with detected keypoints and confidence scores
        """
        if rotation_count != 0:
            frame = apply_rotation(frame, rotation_count)

        frame_h, frame_w = frame.shape[:2]
        keypoints: np.ndarray | None = None
        confidence: np.ndarray | None = None
        prev_bbox = self._prev_bboxes.get(cam_id)

        # Tier 1: Crop to previous detection
        if prev_bbox is not None:
            keypoints, confidence = self._detect_in_region(frame, *prev_bbox)
            if np.sum(confidence >= self.card.confidence_threshold) == 0:
                keypoints, confidence = None, None

        # Tier 2: Full-frame letterbox
        if keypoints is None:
            keypoints, confidence = self._detect_in_region(frame, 0, 0, frame_w, frame_h)
            if np.sum(confidence >= self.card.confidence_threshold) == 0:
                keypoints, confidence = None, None

        # Tier 3: Sliding window scan
        if keypoints is None:
            best_count = 0
            for x1, y1, x2, y2 in self._scan_positions(frame_w, frame_h):
                kps, conf = self._detect_in_region(frame, x1, y1, x2, y2)
                count = int(np.sum(conf >= self.card.confidence_threshold))
                if count > best_count:
                    best_count = count
                    keypoints, confidence = kps, conf
            # If scan also found nothing, use the last attempt
            if keypoints is None:
                keypoints, confidence = self._detect_in_region(frame, 0, 0, frame_w, frame_h)

        assert confidence is not None  # All paths above assign confidence

        # Update tracking state for this camera
        self._prev_bboxes[cam_id] = self._bbox_from_keypoints(keypoints, confidence, frame_w, frame_h)

        # Unrotate points if needed
        if rotation_count != 0:
            keypoints = unrotate_points(keypoints, rotation_count, frame_w, frame_h)

        # Apply confidence threshold and filter
        mask = confidence >= self.card.confidence_threshold
        filtered_keypoints = keypoints[mask]
        filtered_confidence = confidence[mask]
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
    def wireframe(self) -> "WireFrameView | None":
        return self.card.wireframe

    def cleanup(self) -> None:
        """Release onnxruntime session resources.

        Idempotent — safe to call multiple times (multiple FramePacketStreamers
        share a single tracker instance and each calls cleanup on close).
        """
        self._prev_bboxes.clear()
        if hasattr(self, "session"):
            del self.session
            logger.debug(f"OnnxTracker cleaned up: {self.card.name}")
