# type: ignore  # vendored verbatim — excluded from type checking (see module docstring)
"""Vendored numpy post-processing for MoGe-2 ONNX inference.

Vendored from microsoft/MoGe (https://github.com/microsoft/MoGe), MIT license.
The functions are copied unmodified except for their imports: the original
``recover_focal_shift_numpy`` (in ``moge/utils/geometry_numpy.py``) called
``utils3d.np.masked_nearest_resize``; the ``utils3d`` numpy helpers it depends on
(``masked_nearest_resize``, ``uv_map``, ``pixel_coord_map``, ``sliding_window``,
from Ruicheng/utils3d, also MIT) are inlined here so the one cross-package call
resolves locally. No numerical logic was changed.

``recover_focal_shift_numpy`` recovers the two degrees of freedom the MoGe
network's affine-invariant point map cannot fix — focal length and z-shift —
by least-squares over the masked point map.
"""

# ruff: noqa: E501 — vendored verbatim; upstream long docstring/comment lines kept as-is.

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Optional, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    # Unpack lives in typing only on 3.11+; caliscope supports 3.10. Runtime
    # never evaluates the annotation (see ``from __future__ import annotations``),
    # so the checker-only import keeps the vendored signature verbatim.
    from typing_extensions import Unpack


# --------------------------------------------------------------------------- #
# From Ruicheng/utils3d — utils3d/numpy/utils.py
# --------------------------------------------------------------------------- #


def sliding_window(
    x: ndarray,
    window_size: Union[int, Tuple[int, ...]],
    stride: Optional[Union[int, Tuple[int, ...]]] = None,
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None,
    pad_mode: str = "constant",
    pad_value: float = 0,
    axis: Optional[Tuple[int, ...]] = None,
) -> ndarray:
    """
    Get a sliding window of the input array. Window axis(axes) will be appended as the last dimension(s).
    This function is a wrapper of `numpy.lib.stride_tricks.sliding_window_view` with additional support for padding and stride.
    """
    # Process axis
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(window_size, int):
        window_size = (window_size,) * len(axis)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, int):
            pad_size = ((pad_size, pad_size),) * len(axis)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, int) for p in pad_size):
            pad_size = (pad_size,) * len(axis)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(axis)
            else:
                assert len(pad_size) == len(axis), f"pad_size {pad_size} must match the number of axes {len(axis)}"
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in axis else pad_size[axis.index(i)] for i in range(x.ndim)]
        if pad_mode == "constant":
            x = np.pad(x, full_pad, mode=pad_mode, constant_values=pad_value)
        else:
            x = np.pad(x, full_pad, mode=pad_mode)

    # Apply sliding window
    x = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=axis)

    # Apply stride if needed
    if stride is not None:
        if isinstance(stride, int):
            stride = (stride,) * len(axis)
        stride_slice = tuple(
            slice(None) if i not in axis else slice(None, None, stride[axis.index(i)]) for i in range(x.ndim)
        )
        x = x[stride_slice]

    return x


# --------------------------------------------------------------------------- #
# From Ruicheng/utils3d — utils3d/numpy/maps.py
# --------------------------------------------------------------------------- #


def uv_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 0.0,
    left: float = 0.0,
    bottom: float = 1.0,
    right: float = 1.0,
    dtype: np.dtype = np.float32,
) -> ndarray:
    """
    Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.
    This is commonly used as normalized image coordinates in texture mapping (when image is not flipped vertically).
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = np.linspace(left + 0.5 / width, right - 0.5 / width, width, dtype=dtype)
    v = np.linspace(top + 0.5 / height, bottom - 0.5 / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing="xy")
    return np.stack([u, v], axis=2)


def pixel_coord_map(
    *size: Union[int, Tuple[int, int]],
    top: int = 0,
    left: int = 0,
    convention: str = "integer-center",
    dtype: np.dtype = np.float32,
) -> ndarray:
    """
    Get image pixel coordinates map, where (0, 0) is the top-left corner of the top-left pixel, and (width, height) is the bottom-right corner of the bottom-right pixel.
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = np.arange(left, left + width, dtype=dtype)
    v = np.arange(top, top + height, dtype=dtype)
    if convention == "integer-corner":
        assert np.issubdtype(dtype, np.floating), (
            "dtype should be a floating point type when convention is 'integer-corner'"
        )
        u = u + 0.5
        v = v + 0.5
    u, v = np.meshgrid(u, v, indexing="xy")
    return np.stack([u, v], axis=2)


def masked_nearest_resize(
    *image: ndarray,
    mask: ndarray,
    size: Tuple[int, int],
    return_index: bool = False,
) -> Tuple[Unpack[Tuple[ndarray, ...]], ndarray, Tuple[ndarray, ...]]:
    """
    Resize image(s) by nearest sampling with mask awareness.
    """
    height, width = mask.shape[-2:]
    target_height, target_width = size
    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    filter_shape = (filter_h_i, filter_w_i)
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    padding_shape = ((padding_h, padding_h), (padding_w, padding_w))

    # Window the original mask and uv
    pixels = pixel_coord_map(height, width, convention="integer-corner", dtype=np.float32)
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    window_pixels = sliding_window(pixels, window_size=filter_shape, pad_size=padding_shape, axis=(0, 1))
    window_indices = sliding_window(indices, window_size=filter_shape, pad_size=padding_shape, axis=(0, 1))
    window_mask = sliding_window(mask, window_size=filter_shape, pad_size=padding_shape, axis=(-2, -1))

    # Gather the target pixels's local window
    target_centers = uv_map(target_height, target_width, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    target_lefttop = target_centers - np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_window = np.round(target_lefttop).astype(np.int32) + np.array((padding_w, padding_h), dtype=np.int32)

    target_window_pixels = window_pixels[target_window[..., 1], target_window[..., 0], :, :, :].reshape(
        target_height, target_width, 2, filter_size
    )  # (target_height, tgt_width, 2, filter_size)
    target_window_mask = window_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(
        *mask.shape[:-2], target_height, target_width, filter_size
    )  # (..., target_height, tgt_width, filter_size)
    target_window_indices = window_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(
        target_height, target_width, filter_size
    )  # (target_height, tgt_width, filter_size)

    # Compute nearest neighbor in the local window for each pixel
    dist = np.square(target_window_pixels - target_centers[..., None])
    dist = dist[..., 0, :] + dist[..., 1, :]
    dist = np.where(target_window_mask, dist, np.inf)  # (..., target_height, tgt_width, filter_size)
    nearest_in_window = np.argmin(dist, axis=-1, keepdims=True)  # (..., target_height, tgt_width, 1)
    nearest_idx = np.take_along_axis(
        np.broadcast_to(target_window_indices, dist.shape),
        nearest_in_window,
        axis=-1,
    ).squeeze(-1)  # (..., target_height, tgt_width)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = np.any(target_window_mask, axis=-1)
    batch_indices = [
        np.arange(n).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(mask.shape[:-2])
    ]

    nearest_indices = (*batch_indices, nearest_i, nearest_j)
    outputs = tuple(x[nearest_indices] for x in image)

    if return_index:
        return *outputs, target_mask, nearest_indices
    else:
        return *outputs, target_mask


# --------------------------------------------------------------------------- #
# From microsoft/MoGe — moge/utils/geometry_numpy.py
# --------------------------------------------------------------------------- #


def normalized_view_plane_uv_numpy(
    width: int, height: int, aspect_ratio: float = None, dtype: np.dtype = np.float32
) -> np.ndarray:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5

    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing="xy")
    uv = np.stack([u, v], axis=-1)
    return uv


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    return optim_shift


def recover_focal_shift_numpy(
    points: np.ndarray, mask: np.ndarray = None, focal: float = None, downsample_size: Tuple[int, int] = (64, 64)
):
    assert points.shape[-1] == 3, "Points should (H, W, 3)"

    height, width = points.shape[-3], points.shape[-2]

    uv = normalized_view_plane_uv_numpy(width=width, height=height)

    if mask is None:
        points_lr = cv2.resize(points, downsample_size, interpolation=cv2.INTER_LINEAR).reshape(-1, 3)
        uv_lr = cv2.resize(uv, downsample_size, interpolation=cv2.INTER_LINEAR).reshape(-1, 2)
    else:
        points_lr, uv_lr, mask_lr = masked_nearest_resize(points, uv, mask=mask, size=downsample_size)

    if points_lr.size < 2:
        return 1.0, 0.0

    if focal is None:
        shift, focal = solve_optimal_focal_shift(uv_lr, points_lr)
    else:
        shift = solve_optimal_shift(uv_lr, points_lr, focal)

    return focal, shift
