"""ONNX-based pose tracker with generic model card system.

Supports two model output formats:
- SimCC (Simulated Coordinate Classification): 1D probability vectors
- Heatmap: 2D spatial probability maps

Model configuration is loaded from a TOML "model card" file. A card that names
a detector runs top-down: the detector finds the largest person in each frame,
and pose runs on a crop around that box. No box means no keypoints. A card
without a detector runs pose once on the full frame.
"""

import logging

import cv2
import numpy as np

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

        # Lazy import: onnxruntime is only required when constructing a tracker,
        # not for importing this module. This keeps the module import-safe on a
        # lean install (caliscope without the [tracking] extra).
        try:
            import onnxruntime  # type: ignore[reportMissingImports]  # noqa: F401  # no type stubs
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Neural pose tracking requires onnxruntime, which is not installed.\n"
                "Install the tracking extra:\n"
                "    pip install caliscope[tracking]\n"
                "(GUI users: pip install caliscope[gui] includes tracking.)"
            ) from e

        from caliscope.onnx_session import create_inference_session

        logger.info(f"Loading ONNX model: {card.model_path}")
        self.session = create_inference_session(card.model_path)

        # Get input name from model
        self.input_name = self.session.get_inputs()[0].name

        self.detector = None
        if card.detector is not None:
            from caliscope.trackers.person_detector import PersonDetector

            self.detector = PersonDetector(card.detector)

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

    def _pose_region(
        self, box: tuple[float, float, float, float], frame_w: int, frame_h: int, padding: float = 0.125
    ) -> tuple[int, int, int, int]:
        """Expand a detector box into the crop the pose model sees.

        Pads each side by a fraction of the box size (0.125 per side is the
        1.25 box scale RTMPose was trained with), widens or heightens it to the
        model's aspect ratio (3:4 portrait for RTMPose), and clamps to the frame.
        """
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        x_min -= w * padding
        y_min -= h * padding
        x_max += w * padding
        y_max += h * padding

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
        return (x1, y1, x2, y2)

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

    def keypoints_in_box(
        self, frame: np.ndarray, box: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run pose on the crop around one person box, in an unrotated BGR frame.

        Returns all keypoints (N, 2) in frame pixels and their confidences (N,),
        before the confidence threshold is applied.
        """
        frame_h, frame_w = frame.shape[:2]
        return self._detect_in_region(frame, *self._pose_region(box, frame_w, frame_h))

    def _detect(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        """Detect pose keypoints in one frame.

        With a detector, pose runs on a crop around the largest person box and
        an empty PointPacket is returned when no person is found. Without one,
        pose runs on the full frame. Nothing carries over between frames.

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
        if self.detector is None:
            region = (0, 0, frame_w, frame_h)
        else:
            box = self.detector.largest_person(frame)
            if box is None:
                return PointPacket(
                    object_id=np.array([], dtype=np.int32),
                    keypoint_id=np.array([], dtype=np.int32),
                    img_loc=np.empty((0, 2), dtype=np.float32),
                    obj_loc=None,
                    confidence=np.array([], dtype=np.float32),
                )
            region = self._pose_region(box, frame_w, frame_h)

        keypoints, confidence = self._detect_in_region(frame, *region)

        # Unrotate points if needed
        if rotation_count != 0:
            keypoints = unrotate_points(keypoints, rotation_count, frame_w, frame_h)

        # Apply confidence threshold and filter
        mask = confidence >= self.card.confidence_threshold
        filtered_keypoints = keypoints[mask]
        filtered_confidence = confidence[mask]
        keypoint_ids = np.arange(len(confidence), dtype=np.int32)[mask]
        object_ids = np.zeros_like(keypoint_ids)

        return PointPacket(
            object_id=object_ids,
            keypoint_id=keypoint_ids,
            img_loc=filtered_keypoints,
            obj_loc=None,
            confidence=filtered_confidence,
        )

    def get_point_name(self, keypoint_id: int) -> str:
        return self.card.keypoint_id_to_name.get(keypoint_id, str(keypoint_id))

    def scatter_draw_instructions(self, keypoint_id: int) -> dict:
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
        if hasattr(self, "session"):
            del self.session
            logger.debug(f"OnnxTracker cleaned up: {self.card.name}")
        if self.detector is not None and hasattr(self.detector, "session"):
            del self.detector.session
