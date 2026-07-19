"""Per-camera vertical (up-vector) estimation via the GeoCalib field net.

Runs the GeoCalib perspective-field ONNX (up + latitude fields with
confidences) on a handful of frames sampled evenly across each camera's video,
fits gravity per frame with the numpy LM solver (``vertical_solver``), and
aggregates to one up vector per camera plus a frame-to-frame spread diagnostic.

The estimator eats video and emits a :class:`VerticalEstimate` observation
type; it never touches camera state. Focal priors come from each camera's
intrinsic matrix (the solver holds focal fixed and solves the 2-DOF gravity
direction only). Cross-camera disagreement -- the real accuracy signal -- is
computed later by ``CaptureVolume.oriented()`` consumers, not here.

Preprocessing is a cv2/numpy port of GeoCalib's kornia image processor
(``ImagePreprocessor({"resize": 320, "edge_divisible_by": 32})``): resize the
short side to 320 preserving aspect, then center-crop both edges to a multiple
of 32. onnxruntime is a lazy import so this module stays importable on a lean
install; only running the field net requires the ``[tracking]`` extra.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraArray
from caliscope.estimators.model_store import EstimatorModelSpec, ensure_model
from caliscope.estimators.vertical_solver import (
    fit_gravity,
    gravity_vec_from_roll_pitch,
)
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.video_utils import read_video_properties

logger = logging.getLogger(__name__)

# The four dense fields the network emits, in the ONNX graph's output order.
FIELD_NAMES = ("up_field", "up_confidence", "latitude_field", "latitude_confidence")

# GeoCalib's fixed preprocessing geometry.
NET_SHORT_SIDE = 320
EDGE_MULTIPLE = 32

# Below this resize scale (shrinking more than 2x) cv2.INTER_AREA area-averaging
# stands in for kornia's antialiased downsample; above it plain bilinear.
_ANTIALIAS_SCALE_THRESHOLD = 0.5

GEOCALIB_FIELDS_SPEC = EstimatorModelSpec(
    name="geocalib-fields",
    filename="geocalib_fields.onnx",
    source_url=("https://huggingface.co/mprib/geocalib-fields-onnx/resolve/main/geocalib_fields.onnx"),
    sha256="a724447e1bf7138352e5e70bbe0f011e6fb02909edd3148a5d0f46e8d153ae7b",
    extraction="direct",
    file_size_mb=118.2,
    license_info="Apache-2.0",
)


@dataclass(frozen=True)
class VerticalEstimate:
    """Per-camera up vectors with a frame-to-frame stability diagnostic.

    ``up_per_cam`` maps cam_id to a unit up vector in that camera's frame
    (OpenCV convention, ``[0, -1, 0]`` for a level camera), the normalized mean
    over the sampled frames. ``spread_per_cam`` maps cam_id to the median
    angular deviation (degrees) of the per-frame up vectors from that mean --
    small (0.1-0.3 deg on real footage) when the network is stable across the
    clip. It is a stability check, not an accuracy one; cross-camera agreement
    is the accuracy signal and is computed downstream.
    """

    up_per_cam: dict[int, NDArray]
    spread_per_cam: dict[int, float]


def sample_frame_indices(num_frames: int, num_samples: int) -> tuple[int, ...]:
    """Evenly spaced frame indices covering the whole clip, first and last included.

    A clip shorter than ``num_samples`` returns every frame. Indices are unique
    and ascending (np.unique guards the rounding collisions linspace can produce
    when ``num_samples`` approaches ``num_frames``).
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_samples >= num_frames:
        return tuple(range(num_frames))
    spaced = np.linspace(0, num_frames - 1, num_samples).round().astype(int)
    return tuple(int(i) for i in np.unique(spaced))


def _net_size(orig_h: int, orig_w: int) -> tuple[int, int]:
    """(net_h, net_w) with the short side at 320, aspect preserved by truncation.

    Mirrors GeoCalib's ``get_new_image_size(side="short")`` including its int()
    truncation of the long side.
    """
    aspect = orig_w / orig_h
    if aspect < 1.0:  # portrait: width is the short side
        return int(NET_SHORT_SIDE / aspect), NET_SHORT_SIDE
    return NET_SHORT_SIDE, int(NET_SHORT_SIDE * aspect)


def preprocess_frame(frame_bgr: NDArray) -> tuple[NDArray, float, float]:
    """Resize/crop a BGR frame to the field net's input; return it with resize scales.

    Returns ``(image, scale_x, scale_y)`` where ``image`` is a
    ``(1, 3, H, W)`` float32 RGB tensor in [0, 1] with both spatial edges
    divisible by 32, and the scales (net-resolution / original, taken before the
    divisibility crop) map an original-image focal into net pixels for the
    solver. Matches GeoCalib's normalize -> resize -> center-crop order.
    """
    orig_h, orig_w = frame_bgr.shape[:2]
    net_h, net_w = _net_size(orig_h, orig_w)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    scale = min(net_w / orig_w, net_h / orig_h)
    interpolation = cv2.INTER_AREA if scale <= _ANTIALIAS_SCALE_THRESHOLD else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (net_w, net_h), interpolation=interpolation)

    # Scales are net-resolution / original, measured before the crop (the crop
    # shifts the principal point but not the focal), matching GeoCalib.
    scale_x = net_w / orig_w
    scale_y = net_h / orig_h

    # Center-crop to a multiple of 32, matching GeoCalib's fit_to_multiple(crop).
    target_h = (net_h // EDGE_MULTIPLE) * EDGE_MULTIPLE
    target_w = (net_w // EDGE_MULTIPLE) * EDGE_MULTIPLE
    top = -((target_h - net_h) // 2)
    left = -((target_w - net_w) // 2)
    cropped = resized[top : top + target_h, left : left + target_w]

    image = np.ascontiguousarray(cropped.transpose(2, 0, 1)[None])  # (1, 3, H, W)
    return image, scale_x, scale_y


def _load_session(model_path: Path):
    """Create an onnxruntime session, lazily importing onnxruntime."""
    try:
        import onnxruntime  # type: ignore[reportMissingImports]  # noqa: F401  # no type stubs
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Vertical estimation requires onnxruntime, which is not installed.\n"
            "Install the tracking extra:\n"
            "    pip install caliscope[tracking]\n"
            "(GUI users: pip install caliscope[gui] includes tracking.)"
        ) from e

    from caliscope.onnx_session import create_inference_session

    logger.info(f"Loading GeoCalib field net: {model_path}")
    return create_inference_session(model_path)


def _run_field_net(session, image: NDArray, input_name: str) -> tuple[NDArray, ...]:
    """Run the field net; return the four fields in FIELD_NAMES order."""
    outputs = session.run(None, {input_name: image})
    by_name = {out.name: value for out, value in zip(session.get_outputs(), outputs)}
    return tuple(by_name[name] for name in FIELD_NAMES)


def _angle_deg(vec_a: NDArray, vec_b: NDArray) -> float:
    """Angle between two unit vectors in degrees."""
    cosine = float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _up_vectors_for_camera(
    session,
    input_name: str,
    video_path: Path,
    cam_id: int,
    focal_x: float,
    focal_y: float,
    frames_per_camera: int,
) -> list[NDArray]:
    """Sample frames, run the field net + solver on each, return per-frame up vectors."""
    properties = read_video_properties(video_path)
    indices = sample_frame_indices(properties["frame_count"], frames_per_camera)

    ups: list[NDArray] = []
    source = FrameSource.from_path(video_path, cam_id=cam_id, wanted_indices=set(indices))
    try:
        while (packet := source.next_frame()) is not None:
            image, scale_x, scale_y = preprocess_frame(packet.frame)
            fields = _run_field_net(session, image, input_name)
            fit = fit_gravity(
                *fields,
                focal_x_px=focal_x * scale_x,
                focal_y_px=focal_y * scale_y,
            )
            ups.append(gravity_vec_from_roll_pitch(fit.roll_rad, fit.pitch_rad))
    finally:
        source.close()

    if not ups:
        raise ValueError(f"No frames decoded for cam {cam_id} at {video_path}")
    return ups


def _process_camera_vertical(
    session,
    input_name: str,
    cam_id: int,
    video: Path,
    focal_x: float,
    focal_y: float,
    frames_per_camera: int,
) -> tuple[int, NDArray, float, list[NDArray]]:
    """Run vertical estimation for one camera. Returns (cam_id, consensus_up, spread, raw_ups)."""
    ups = _up_vectors_for_camera(session, input_name, video, cam_id, focal_x, focal_y, frames_per_camera)
    stacked = np.array(ups)
    consensus = stacked.mean(axis=0)
    consensus = consensus / np.linalg.norm(consensus)
    spread = float(np.median([_angle_deg(up, consensus) for up in ups]))
    return cam_id, consensus, spread, ups


def estimate_vertical(
    videos: Mapping[int, Path | str],
    cameras: CameraArray,
    *,
    frames_per_camera: int = 12,
) -> VerticalEstimate:
    """Estimate a per-camera up vector from video, using GeoCalib's field net.

    ``videos`` maps cam_id to that camera's video path. Each camera must carry an
    intrinsic ``matrix`` in ``cameras`` (fx, fy supply the fixed focal prior).
    Downloads the field-net weights on first use. Runs ``frames_per_camera``
    evenly spaced frames per camera; the up vector is the normalized mean over
    those frames and the spread is their median angular deviation from it.

    Cameras are processed concurrently. The ONNX session is thread-safe for
    ``run()`` calls; video decoding is per-camera with no shared state.
    """
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    model_path = ensure_model(GEOCALIB_FIELDS_SPEC)
    session = _load_session(model_path)
    input_name = session.get_inputs()[0].name

    cam_args: list[tuple[int, Path, float, float]] = []
    for cam_id, video in videos.items():
        camera = cameras[cam_id]
        if camera.matrix is None:
            raise ValueError(
                f"Camera {cam_id} lacks an intrinsic matrix; vertical estimation "
                "needs a focal prior. Calibrate intrinsics first."
            )
        cam_args.append(
            (
                cam_id,
                Path(video),
                float(camera.matrix[0, 0]),
                float(camera.matrix[1, 1]),
            )
        )

    up_per_cam: dict[int, NDArray] = {}
    spread_per_cam: dict[int, float] = {}

    with ThreadPoolExecutor(max_workers=len(cam_args)) as executor:
        futures = {
            executor.submit(
                _process_camera_vertical,
                session,
                input_name,
                cam_id,
                video,
                fx,
                fy,
                frames_per_camera,
            ): cam_id
            for cam_id, video, fx, fy in cam_args
        }
        for future in concurrent.futures.as_completed(futures):
            cam_id, consensus, spread, ups = future.result()
            up_per_cam[cam_id] = consensus
            spread_per_cam[cam_id] = spread
            logger.info(
                f"cam {cam_id}: up {consensus.round(4).tolist()}, frame spread {spread:.4f} deg over {len(ups)} frames"
            )

    return VerticalEstimate(up_per_cam=up_per_cam, spread_per_cam=spread_per_cam)
