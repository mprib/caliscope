"""MoGe-2 runner: per-camera focal length + metric depth at tracked keypoints.

MoGe-2 (Microsoft) estimates an affine-invariant point map, a validity mask, and
a metric scale from a single image. This runner drives the official ONNX export
over a handful of sampled frames per camera and emits two observation types:

- ``focal_per_cam``: the median per-frame focal length in pixels, a source of
  camera intrinsics for the epipolar bootstrap (M1).
- ``depth_observations``: metric depth sampled at each tracked keypoint, a
  ``DepthObservation`` scale cue for ``CaptureVolume.scaled()`` (M2).

The affine-invariant point map leaves two degrees of freedom the network cannot
know — focal length and z-shift — recovered by the vendored numpy post-processing
in ``moge_utils`` (``recover_focal_shift_numpy``). Depth is then
``(points_z + shift) * metric_scale``.

onnxruntime is imported lazily (it ships in the ``[tracking]`` extra); this
module imports cleanly on a lean install.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from caliscope.core.point_data import ImagePoints
from caliscope.core.scale_cues import DepthObservation
from caliscope.estimators.model_store import EstimatorModelSpec, ensure_model
from caliscope.estimators.moge_utils import recover_focal_shift_numpy
from caliscope.packets import PixelFormat
from caliscope.recording.frame_source import FrameSource

logger = logging.getLogger(__name__)


# num_tokens for the ViT-L backbone (fixed by the model architecture).
MOGE_NUM_TOKENS = 3600

MOGE_MODEL_SPEC = EstimatorModelSpec(
    name="moge-2-vitl-normal",
    filename="moge_2_vitl_normal.onnx",
    source_url="https://huggingface.co/Ruicheng/moge-2-vitl-normal-onnx/resolve/main/model.onnx",
    sha256="afbc4ccc3450298f3afb35b90f015f4c4f552dea21dc6470d5f7b78b77e2d751",
    extraction="direct",
    file_size_mb=1262.9,  # 1324265014 bytes
    license_info="MIT",
)


@dataclass(frozen=True)
class MoGeResult:
    """One MoGe run: focal estimates and metric depths at tracked keypoints."""

    focal_per_cam: dict[int, float]  # cam_id -> focal length in pixels
    depth_observations: list[DepthObservation]


def run_moge(
    videos: Mapping[int, Path | str],
    points: ImagePoints,
    *,
    frames_per_camera: int = 4,
) -> MoGeResult:
    """Run MoGe-2 on sampled frames and estimate focal length + keypoint depths.

    For each camera, up to ``frames_per_camera`` sync indices that carry keypoint
    detections are sampled, spread evenly across the observed range. Each sampled
    frame is decoded once (via PyAV) and run through MoGe. The camera's focal
    length is the median of per-frame estimates; a ``DepthObservation`` is emitted
    for every tracked keypoint whose pixel falls on a valid (masked, positive)
    depth. Observations stay keyed by sync_index (the domain's time key).

    Decoding uses the ``frame_index`` column when present (from
    ``extract_image_points_multicam``): sync_index and frame_index diverge when
    cameras start staggered or drop frames, so decoding by sync_index would sample
    the depth map at the wrong instant. When ``frame_index`` is absent (the
    single-video ``extract_image_points`` path), sync_index is the frame index.
    """
    session = _build_session()
    df = points.df
    has_frame_index = "frame_index" in df.columns

    focal_per_cam: dict[int, float] = {}
    depth_observations: list[DepthObservation] = []

    for cam_id in sorted(df["cam_id"].unique()):
        cam_id = int(cam_id)
        if cam_id not in videos:
            logger.warning(f"cam_id {cam_id} has image points but no video; skipping MoGe run.")
            continue

        cam_df = df[df["cam_id"] == cam_id]
        sync_indices = _sample_sync_indices(cam_df["sync_index"].to_numpy(), frames_per_camera)
        if len(sync_indices) == 0:
            continue

        sync_to_frame = _sync_to_frame_map(cam_df, has_frame_index)
        frame_indices = [sync_to_frame[s] for s in sync_indices]
        frames = _decode_frames(Path(videos[cam_id]), cam_id, frame_indices)

        frame_focals: list[float] = []
        for sync_index in sync_indices:
            rgb = frames.get(sync_to_frame[sync_index])
            if rgb is None:
                logger.warning(f"cam_id {cam_id}: frame {sync_to_frame[sync_index]} not decoded; skipping.")
                continue

            point_map, mask_binary, metric_scale = _infer_frame(session, rgb)

            focal, shift = recover_focal_shift_numpy(point_map, mask_binary)
            frame_focals.append(_focal_to_pixels(float(focal), point_map.shape[1], point_map.shape[0]))

            depth_map = (point_map[..., 2] + shift) * metric_scale
            valid_depth = mask_binary & (depth_map > 0)

            frame_rows = cam_df[cam_df["sync_index"] == sync_index]
            depth_observations.extend(_keypoint_depths(frame_rows, depth_map, valid_depth, cam_id, sync_index))

        if frame_focals:
            focal_per_cam[cam_id] = float(np.median(frame_focals))

    return MoGeResult(focal_per_cam=focal_per_cam, depth_observations=depth_observations)


def _build_session():
    """Construct the MoGe ONNX inference session, downloading the model on first use.

    onnxruntime is imported here (not at module scope) so the module stays
    importable on a lean install without the ``[tracking]`` extra.
    """
    try:
        import onnxruntime  # type: ignore[reportMissingImports]  # noqa: F401  # no type stubs
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "MoGe estimation requires onnxruntime, which is not installed.\n"
            "Install the tracking extra:\n"
            "    pip install caliscope[tracking]\n"
            "(GUI users: pip install caliscope[gui] includes tracking.)"
        ) from e

    model_path = ensure_model(MOGE_MODEL_SPEC)
    logger.info(f"Loading MoGe ONNX model: {model_path}")
    from caliscope.onnx_session import create_inference_session

    return create_inference_session(model_path)


def _infer_frame(session, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one RGB frame through MoGe, returning (point_map, mask_binary, metric_scale).

    ``point_map`` is (H, W, 3) affine-invariant; ``mask_binary`` is (H, W) bool
    (sigmoid > 0.5); ``metric_scale`` is the scalar that maps shifted z to meters.
    """
    image = (rgb.transpose(2, 0, 1)[None].astype(np.float32)) / 255.0

    feed: dict[str, np.ndarray] = {}
    for model_input in session.get_inputs():
        if model_input.name == "image":
            feed["image"] = image
        elif model_input.name == "num_tokens":
            feed["num_tokens"] = np.array(MOGE_NUM_TOKENS, dtype=np.int64)
        else:
            raise ValueError(f"unexpected MoGe ONNX input: {model_input.name}")

    output_names = [o.name for o in session.get_outputs()]
    raw = dict(zip(output_names, session.run(None, feed)))

    point_map = raw["points"][0].astype(np.float32)  # (H, W, 3)
    mask_binary = raw["mask"][0] > 0.5
    metric_scale = float(raw["metric_scale"][0])
    return point_map, mask_binary, metric_scale


def _focal_to_pixels(focal: float, width: int, height: int) -> float:
    """Convert MoGe's recovered focal (in normalized-diagonal units) to pixels.

    ``recover_focal_shift_numpy`` works in MoGe's normalized view-plane UV, where
    the image diagonal spans 2 units. Converting to the utils3d "normalized
    intrinsics" convention (fx over image width) and then to pixels:

        fx_norm = focal / 2 * sqrt(1 + aspect^2) / aspect
        focal_px = fx_norm * width

    Matches monokin's validated probe (probe_moge_onnx.py).
    """
    aspect_ratio = width / height
    fx_norm = focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
    return fx_norm * width


def _keypoint_depths(
    frame_rows,
    depth_map: np.ndarray,
    valid_depth: np.ndarray,
    cam_id: int,
    sync_index: int,
) -> list[DepthObservation]:
    """Sample metric depth at each keypoint pixel, skipping invalid (unmasked) samples."""
    height, width = depth_map.shape
    observations: list[DepthObservation] = []
    for row in frame_rows.itertuples(index=False):
        px = int(round(row.img_loc_x))
        py = int(round(row.img_loc_y))
        px = min(max(px, 0), width - 1)
        py = min(max(py, 0), height - 1)
        if not valid_depth[py, px]:
            continue
        observations.append(
            DepthObservation(
                cam_id=cam_id,
                keypoint_id=int(row.keypoint_id),
                sync_index=sync_index,
                depth_m=float(depth_map[py, px]),
            )
        )
    return observations


def _sync_to_frame_map(cam_df, has_frame_index: bool) -> dict[int, int]:
    """Map each sync_index to the video frame index to decode for it.

    With a frame_index column, the mapping comes from the data (frame_index is
    constant within a (cam_id, sync_index) group). Without it, sync_index is the
    frame index — the identity map.
    """
    if not has_frame_index:
        return {int(s): int(s) for s in cam_df["sync_index"].unique()}
    deduped = cam_df.drop_duplicates("sync_index")
    return {int(s): int(f) for s, f in zip(deduped["sync_index"], deduped["frame_index"])}


def _sample_sync_indices(sync_indices: np.ndarray, count: int) -> list[int]:
    """Pick up to ``count`` unique sync indices spread evenly across the observed range."""
    unique = np.unique(sync_indices)
    if len(unique) <= count:
        return [int(s) for s in unique]
    positions = np.linspace(0, len(unique) - 1, count).round().astype(int)
    return [int(unique[p]) for p in np.unique(positions)]


def _decode_frames(video_path: Path, cam_id: int, sync_indices: list[int]) -> dict[int, np.ndarray]:
    """Decode the requested frames as RGB, keyed by frame index (== sync_index).

    MoGe expects RGB; ``FrameSource`` yields BGR, so channels are flipped here.
    """
    wanted = set(sync_indices)
    frames: dict[int, np.ndarray] = {}
    with FrameSource.from_path(
        video_path, cam_id=cam_id, wanted_indices=wanted, pixel_format=PixelFormat.BGR
    ) as source:
        while (packet := source.next_frame()) is not None:
            frames[packet.frame_index] = np.ascontiguousarray(packet.frame[..., ::-1])
    return frames
