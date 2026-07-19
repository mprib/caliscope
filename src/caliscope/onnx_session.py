"""Shared onnxruntime session factory with automatic provider selection.

All caliscope inference (pose trackers, GeoCalib) runs on onnxruntime.
Sessions prefer CUDA when the installed onnxruntime build exposes it (the
onnxruntime-gpu package on a CUDA machine) and fall back to CPU otherwise.
No configuration: the installed package decides.

Callers lazy-import onnxruntime themselves first so their error messages can
name the right install extra; this module assumes it is importable.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_inference_session(model_path: Path):
    """Create an onnxruntime InferenceSession on the best available provider.

    Prefers CUDAExecutionProvider when the installed build advertises it,
    with CPUExecutionProvider always appended as fallback. onnxruntime-gpu
    ships CUDA/cuDNN via nvidia-* wheels but does not load them on its own;
    preload_dlls() makes them visible, without which the CUDA provider
    silently falls back to CPU.
    """
    import onnxruntime as ort  # type: ignore[reportMissingImports]  # no type stubs

    cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
    if cuda_available and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    providers: list = []
    if cuda_available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), providers=providers)

    active = session.get_providers()[0]
    if cuda_available and active != "CUDAExecutionProvider":
        logger.warning(
            f"CUDAExecutionProvider is installed but did not bind for {model_path.name}; "
            f"running on {active}. Check the CUDA/cuDNN runtime."
        )
    else:
        logger.info(f"ONNX session for {model_path.name} on {active}")
    return session
