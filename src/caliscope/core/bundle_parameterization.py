from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from caliscope.cameras.camera_array import CameraArray
from caliscope.exceptions import CalibrationError

N_EXTRINSIC_PARAMS = 6
N_FREE_INTRINSIC_PARAMS = 3


@dataclass(frozen=True)
class BoundWarning:
    cam_id: int
    parameter: str  # "f" | "k1" | "k2"
    bound: str  # "lower" | "upper"
    value: float


@dataclass(frozen=True)
class IntrinsicEstimate:
    cam_id: int
    f_recovered: float
    k1_recovered: float
    k2_recovered: float
    f_initial: float
    k1_initial: float
    k2_initial: float


@dataclass(frozen=True)
class CameraBlock:
    cam_id: int
    free_intrinsics: bool
    fx_initial: float
    fy_initial: float
    cx: float
    cy: float
    fisheye: bool
    dist_fixed: tuple[float, ...]
    k1_initial: float = 0.0
    k2_initial: float = 0.0

    @property
    def n_params(self) -> int:
        return N_EXTRINSIC_PARAMS + N_FREE_INTRINSIC_PARAMS if self.free_intrinsics else N_EXTRINSIC_PARAMS


@dataclass(frozen=True)
class BundleParameterization:
    blocks: tuple[CameraBlock, ...]
    n_points: int

    @classmethod
    def from_camera_array(
        cls, camera_array: CameraArray, n_points: int, *, refine_intrinsics: bool
    ) -> BundleParameterization:
        blocks: list[CameraBlock] = []
        for idx in sorted(camera_array.posed_index_to_cam_id.keys()):
            cam_id = camera_array.posed_index_to_cam_id[idx]
            cam = camera_array.cameras[cam_id]

            if cam.matrix is None or cam.distortions is None:
                raise CalibrationError(
                    f"Camera {cam_id} has no intrinsics. "
                    f"Run intrinsic calibration or synthesize defaults before optimizing."
                )

            dist = cam.distortions.ravel()

            if cam.fisheye:
                n = len(dist)
                if n != 4:
                    raise CalibrationError(
                        f"Fisheye camera {cam_id} requires exactly 4 distortion coefficients "
                        f"(equidistant model), got {n}."
                    )
                blocks.append(
                    CameraBlock(
                        cam_id=cam_id,
                        free_intrinsics=False,
                        fx_initial=float(cam.matrix[0, 0]),
                        fy_initial=float(cam.matrix[1, 1]),
                        cx=float(cam.matrix[0, 2]),
                        cy=float(cam.matrix[1, 2]),
                        fisheye=True,
                        dist_fixed=tuple(dist),
                    )
                )
            else:
                # Brown-Conrady: dist = [k1, k2, p1, p2, k3]
                blocks.append(
                    CameraBlock(
                        cam_id=cam_id,
                        free_intrinsics=refine_intrinsics,
                        fx_initial=float(cam.matrix[0, 0]),
                        fy_initial=float(cam.matrix[1, 1]),
                        cx=float(cam.matrix[0, 2]),
                        cy=float(cam.matrix[1, 2]),
                        fisheye=False,
                        dist_fixed=tuple(dist[2:5]),
                        k1_initial=float(dist[0]),
                        k2_initial=float(dist[1]),
                    )
                )

        return cls(blocks=tuple(blocks), n_points=n_points)

    @cached_property
    def camera_param_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        running = 0
        for block in self.blocks:
            offsets.append(running)
            running += block.n_params
        return tuple(offsets)

    @cached_property
    def n_camera_params(self) -> int:
        return sum(b.n_params for b in self.blocks)

    def pack(self, camera_array: CameraArray, world_points_xyz: NDArray) -> NDArray:
        parts: list[NDArray] = []
        for block in self.blocks:
            cam = camera_array.cameras[block.cam_id]
            parts.append(cam.extrinsics_to_vector())
            if block.free_intrinsics:
                dist = cam.distortions.ravel()  # type: ignore[union-attr]
                parts.append(np.array([1.0, float(dist[0]), float(dist[1])]))
        parts.append(world_points_xyz.ravel())
        return np.concatenate(parts)

    def unpack_into(self, camera_array: CameraArray, x: NDArray) -> NDArray:
        for i, block in enumerate(self.blocks):
            off = self.camera_param_offsets[i]
            cam = camera_array.cameras[block.cam_id]
            cam.extrinsics_from_vector(x[off : off + 6])
            if block.free_intrinsics:
                s, k1, k2 = x[off + 6 : off + 9]
                cam.matrix = np.array(
                    [[s * block.fx_initial, 0.0, block.cx], [0.0, s * block.fy_initial, block.cy], [0.0, 0.0, 1.0]]
                )
                cam.distortions = np.array([k1, k2, *block.dist_fixed])
        return x[self.n_camera_params :].reshape(-1, 3)

    def bounds(self) -> tuple[NDArray, NDArray]:
        n_total = self.n_camera_params + 3 * self.n_points
        lower = np.full(n_total, -np.inf)
        upper = np.full(n_total, np.inf)
        for i, block in enumerate(self.blocks):
            if block.free_intrinsics:
                off = self.camera_param_offsets[i] + 6
                lower[off] = 0.5
                upper[off] = 2.0
                lower[off + 1] = -1.0
                upper[off + 1] = 1.0
                lower[off + 2] = -2.0
                upper[off + 2] = 2.0
        return lower, upper

    def trial_projection_inputs(self, x: NDArray, block_index: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        block = self.blocks[block_index]
        off = self.camera_param_offsets[block_index]
        rvec = x[off : off + 3]
        tvec = x[off + 3 : off + 6]

        if block.free_intrinsics:
            s, k1, k2 = x[off + 6 : off + 9]
            K = np.array(
                [[s * block.fx_initial, 0.0, block.cx], [0.0, s * block.fy_initial, block.cy], [0.0, 0.0, 1.0]]
            )
            dist = np.array([k1, k2, *block.dist_fixed])
        elif block.fisheye:
            K = np.array([[block.fx_initial, 0.0, block.cx], [0.0, block.fy_initial, block.cy], [0.0, 0.0, 1.0]])
            dist = np.array(block.dist_fixed)
        else:
            # Locked Brown-Conrady: reconstruct full [k1, k2, p1, p2, k3]
            K = np.array([[block.fx_initial, 0.0, block.cx], [0.0, block.fy_initial, block.cy], [0.0, 0.0, 1.0]])
            dist = np.array([block.k1_initial, block.k2_initial, *block.dist_fixed])

        return rvec, tvec, K, dist

    def sparsity(
        self,
        camera_indices: NDArray,
        obj_indices: NDArray,
        n_constraints: int,
        constraint_groups_a: NDArray | None,
        constraint_groups_b: NDArray | None,
    ) -> lil_matrix:
        n_observations = len(camera_indices)
        n_residuals = n_observations * 2 + n_constraints
        n_params = self.n_camera_params + 3 * self.n_points

        sp = lil_matrix((n_residuals, n_params), dtype=int)
        obs_idx = np.arange(n_observations)

        for i, block in enumerate(self.blocks):
            cam_mask = camera_indices == i
            if not cam_mask.any():
                continue
            obs_for_cam = obs_idx[cam_mask]
            off = self.camera_param_offsets[i]
            for p in range(block.n_params):
                sp[2 * obs_for_cam, off + p] = 1
                sp[2 * obs_for_cam + 1, off + p] = 1

        for coord in range(3):
            param_col = self.n_camera_params + obj_indices * 3 + coord
            sp[2 * obs_idx, param_col] = 1
            sp[2 * obs_idx + 1, param_col] = 1

        if constraint_groups_a is not None and constraint_groups_b is not None and n_constraints > 0:
            # Each constraint row depends on the 3 coordinate columns of every
            # world-point row in both width-4 endpoint groups (up to 24 columns).
            # Corner groups repeat one row 4x; re-marking a column is idempotent.
            c_idx = np.arange(n_constraints)
            row_offset = n_observations * 2
            for groups in (constraint_groups_a, constraint_groups_b):
                for col in range(groups.shape[1]):
                    for coord in range(3):
                        param_col = self.n_camera_params + groups[:, col] * 3 + coord
                        sp[row_offset + c_idx, param_col] = 1

        return sp

    def bound_warnings(self, x: NDArray) -> tuple[BoundWarning, ...]:
        warnings: list[BoundWarning] = []
        for i, block in enumerate(self.blocks):
            if not block.free_intrinsics:
                continue
            off = self.camera_param_offsets[i] + 6
            s = float(x[off])
            k1 = float(x[off + 1])
            k2 = float(x[off + 2])

            # s bounds: [0.5, 2.0], proximity = 1% relative
            if abs(s - 0.5) <= 0.01 * 0.5:
                warnings.append(BoundWarning(block.cam_id, "f", "lower", s * block.fx_initial))
            if abs(s - 2.0) <= 0.01 * 2.0:
                warnings.append(BoundWarning(block.cam_id, "f", "upper", s * block.fx_initial))

            # k1 bounds: [-1, 1], proximity = absolute 0.01
            if abs(k1 - (-1.0)) <= 0.01:
                warnings.append(BoundWarning(block.cam_id, "k1", "lower", k1))
            if abs(k1 - 1.0) <= 0.01:
                warnings.append(BoundWarning(block.cam_id, "k1", "upper", k1))

            # k2 bounds: [-2, 2], proximity = absolute 0.01
            if abs(k2 - (-2.0)) <= 0.01:
                warnings.append(BoundWarning(block.cam_id, "k2", "lower", k2))
            if abs(k2 - 2.0) <= 0.01:
                warnings.append(BoundWarning(block.cam_id, "k2", "upper", k2))

        return tuple(warnings)

    def intrinsic_estimates(self, camera_array: CameraArray) -> tuple[IntrinsicEstimate, ...]:
        estimates: list[IntrinsicEstimate] = []
        for block in self.blocks:
            if not block.free_intrinsics:
                continue
            cam = camera_array.cameras[block.cam_id]
            estimates.append(
                IntrinsicEstimate(
                    cam_id=block.cam_id,
                    f_recovered=float(cam.matrix[0, 0]),  # type: ignore[index]
                    k1_recovered=float(cam.distortions[0]),  # type: ignore[index]
                    k2_recovered=float(cam.distortions[1]),  # type: ignore[index]
                    f_initial=block.fx_initial,
                    k1_initial=block.k1_initial,
                    k2_initial=block.k2_initial,
                )
            )
        return tuple(estimates)
