"""Torch-free LM solver: fit roll and pitch to GeoCalib's perspective fields.

A numpy port of GeoCalib's LM optimizer (ETH CVG, ECCV 2024, Apache 2.0;
github.com/cvg/GeoCalib, lm_optimizer.py / perspective_fields.py / gravity.py /
misc.py), reduced to the one case the pipeline needs: pinhole camera, focal
known and fixed, solve the 2-DOF gravity direction. Together with the field-net
ONNX export this removes torch from gravity estimation.

The solver's forward model: propose a gravity direction, render the up and
latitude fields that camera tilt would produce (closed-form geometry), and
compare against the network's predicted fields, weighting each pixel by the
network's confidence and a Huber robust loss. Levenberg-Marquardt iterates on
the unit sphere (Householder-parameterized tangent steps). Faithful port
quirks, kept deliberately: every step is accepted (no rollback -- a cost
increase only raises the damping lambda), latitude residuals live in sine
space, and the second residual evaluation per iteration drives both the lambda
update and early stopping.

Inputs are the raw field-net outputs at net resolution and the focal in
net-resolution pixels. Principal point is assumed at the image center, matching
GeoCalib's pinhole configuration.

Ported verbatim from monokin (src/monokin/gravity_solver.py), validated to
1.6e-5 deg parity against torch GeoCalib on identical fields. Numerics are kept
identical; only the module docstring notes the provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DEFAULT_NUM_STEPS = 30
DEFAULT_INITIAL_LAMBDA = 0.1
LAMBDA_MIN, LAMBDA_MAX = 1e-6, 1e2
STOP_ATOL = STOP_RTOL = 1e-8
UP_LOSS_SCALE = LAT_LOSS_SCALE = 1e-2


@dataclass(frozen=True)
class GravityFit:
    """The solver's result: gravity angles, Hessian uncertainties, run evidence."""

    roll_rad: float
    pitch_rad: float
    roll_uncertainty_rad: float
    pitch_uncertainty_rad: float
    gravity_uncertainty_rad: float
    initial_cost: float
    final_cost: float
    stop_step: int


def gravity_vec_from_roll_pitch(roll_rad: float, pitch_rad: float) -> NDArray:
    """Unit up vector in the OpenCV camera frame; [0, -1, 0] for a level camera."""
    sin_roll, cos_roll = np.sin(roll_rad), np.cos(roll_rad)
    sin_pitch, cos_pitch = np.sin(pitch_rad), np.cos(pitch_rad)
    return np.array([-sin_roll * cos_pitch, -cos_roll * cos_pitch, sin_pitch], dtype=np.float64)


def roll_pitch_from_gravity_vec(vec: NDArray) -> tuple[float, float]:
    """Angles back out of the vector; GeoCalib's branch handles roll beyond +/-90."""
    eps = 1e-4  # Gravity.eps
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    pitch = float(np.arcsin(np.clip(z, -1.0, 1.0)))
    roll = float(np.arcsin(np.clip(-x / (np.sqrt(max(1 - z**2, 0.0)) + eps), -1.0, 1.0)))
    if y >= 0:  # upside-down camera: reflect and offset
        roll = -roll - np.pi * np.sign(x)
    return roll, pitch


def _householder_vector(x: NDArray) -> tuple[NDArray, float]:
    """v (v[-1] = 1) and beta with (I - beta v v^T) x = |x| e_n; last-element pivot."""
    sigma = float(np.sum(x[:-1] ** 2))
    x_pivot = float(x[-1])
    norm = float(np.linalg.norm(x))
    sigma = max(sigma, 1e-7)
    v_pivot = x_pivot - norm if x_pivot < 0 else -sigma / (x_pivot + norm)
    beta = 2 * v_pivot**2 / (sigma + v_pivot**2)
    v = np.concatenate([x[:-1] / v_pivot, [1.0]])
    return v, beta


def spherical_plus(x: NDArray, delta: NDArray) -> NDArray:
    """Move on the unit sphere: exponential map of the tangent step, then rotate to x."""
    eps = 1e-7
    norm_delta = float(np.linalg.norm(delta))
    sinc = 1.0 if norm_delta < eps else np.sin(norm_delta) / norm_delta
    exp_delta = np.concatenate([sinc * delta, [np.cos(norm_delta)]])
    v, beta = _householder_vector(x)
    householder_applied = exp_delta - v * (beta * float(np.dot(v, exp_delta)))
    return float(np.linalg.norm(x)) * householder_applied


def spherical_j_plus(x: NDArray) -> NDArray:
    """(3, 2) Jacobian of spherical_plus at delta = 0: the tangent basis at x."""
    v, beta = _householder_vector(x)
    householder = np.eye(3) - beta * np.outer(v, v)
    return householder[:, :2]


def j_roll_pitch(vec: NDArray) -> NDArray:
    """(3, 2) Jacobian of the up vector w.r.t. (roll, pitch), for uncertainty only."""
    roll, pitch = roll_pitch_from_gravity_vec(vec)
    sin_roll, cos_roll = np.sin(roll), np.cos(roll)
    sin_pitch, cos_pitch = np.sin(pitch), np.cos(pitch)
    j_roll = np.array([-cos_roll * cos_pitch, sin_roll * cos_pitch, 0.0])
    j_pitch = np.array([sin_roll * sin_pitch, cos_roll * sin_pitch, cos_pitch])
    return np.stack([j_roll, j_pitch], axis=-1)


def _huber(x: NDArray) -> tuple[NDArray, NDArray]:
    """Huber loss and first derivative on already-squared costs (GeoCalib's form)."""
    mask = x <= 1
    sqrt_x = np.sqrt(x + 1e-8)
    inv_sqrt_x = np.maximum(np.finfo(np.float32).eps, 1 / sqrt_x)
    loss = np.where(mask, x, 2 * sqrt_x - 1)
    weight = np.where(mask, 1.0, inv_sqrt_x)
    return loss, weight


def _scaled_huber(squared: NDArray, scale: float) -> tuple[NDArray, NDArray]:
    """GeoCalib's scaled_loss: pre-divide by scale^2, post-multiply the loss."""
    scale2 = scale**2
    loss, weight = _huber(squared / scale2)
    return loss * scale2, weight


class _PerspectiveGeometry:
    """Per-image constants: the pixel grid, its rays, and the field targets."""

    def __init__(
        self,
        up_field: NDArray,
        up_confidence: NDArray,
        latitude_field: NDArray,
        latitude_confidence: NDArray,
        focal_x_px: float,
        focal_y_px: float,
    ) -> None:
        height, width = up_confidence.shape[-2:]
        # Pixel grid in GeoCalib's row-major (h, w) order, origin at pixel (0, 0),
        # principal point at the image center.
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        xy = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)
        center = np.array([width / 2, height / 2])
        focal = np.array([focal_x_px, focal_y_px])
        self.uv = (xy - center) / focal  # (N, 2) normalized image coords

        rays = np.concatenate([self.uv, np.ones((self.uv.shape[0], 1))], axis=-1)
        self.rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)  # (N, 3)

        self.target_up = np.asarray(up_field, dtype=np.float64).reshape(2, -1).T  # (N, 2)
        self.target_sin_lat = np.sin(np.asarray(latitude_field, dtype=np.float64).reshape(-1))
        self.up_confidence = np.asarray(up_confidence, dtype=np.float64).reshape(-1)
        self.lat_confidence = np.asarray(latitude_confidence, dtype=np.float64).reshape(-1)

    def render(self, vec: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Fields a camera tilted to `vec` would produce.

        Returns (up_normalized (N, 2), up_unnormalized (N, 2), sin_latitude (N,)).
        The up vector at a pixel is the image-plane projection of world up:
        (a, b) - c * uv for gravity vec (a, b, c). Latitude's sine is the dot of
        the pixel ray with the up vector -- GeoCalib compares sines, so the
        arcsine never enters the solver.
        """
        up_unnormalized = vec[:2][None, :] - vec[2] * self.uv  # (N, 2)
        norms = np.linalg.norm(up_unnormalized, axis=-1, keepdims=True)
        up_normalized = up_unnormalized / np.maximum(norms, 1e-12)
        sin_latitude = np.clip(self.rays @ vec, -1 + 1e-6, 1 - 1e-6)
        return up_normalized, up_unnormalized, sin_latitude

    def residuals_and_costs(self, vec: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, float]:
        """Residuals (target - rendered) and confidence-weighted Huber costs."""
        up_normalized, _, sin_latitude = self.render(vec)
        up_residual = self.target_up - up_normalized  # (N, 2)
        lat_residual = self.target_sin_lat - sin_latitude  # (N,)

        up_cost, up_weight = _scaled_huber((up_residual**2).sum(axis=-1), UP_LOSS_SCALE)
        lat_cost, lat_weight = _scaled_huber(lat_residual**2, LAT_LOSS_SCALE)
        up_weight = up_weight * self.up_confidence
        lat_weight = lat_weight * self.lat_confidence
        total_cost = float((up_cost * self.up_confidence).mean() + (lat_cost * self.lat_confidence).mean())
        return up_residual, up_weight, lat_residual, lat_weight, total_cost

    def gradient_and_hessian(self, vec: NDArray, tangent_basis: NDArray) -> tuple[NDArray, NDArray]:
        """Confidence-weighted J^T r and J^T J over both fields, 2 parameters.

        tangent_basis (3, 2) maps parameter steps to gravity-vector changes:
        the spherical manifold basis during optimization, d(vec)/d(roll, pitch)
        for the uncertainty readout.
        """
        up_residual, up_weight, lat_residual, lat_weight, _ = self.residuals_and_costs(vec)
        up_normalized, up_unnormalized, _ = self.render(vec)

        # Up-field Jacobian: normalize(...) of a linear map of the gravity vec.
        # d(up_unnorm)/d(abc) at each pixel is [[1, 0, -u], [0, 1, -v]].
        norms = np.maximum(np.linalg.norm(up_unnormalized, axis=-1, keepdims=True)[..., None], 1e-6)
        outer = up_unnormalized[..., :, None] * up_unnormalized[..., None, :]
        j_normalize = np.eye(2)[None] / norms - outer / norms**3  # (N, 2, 2)

        j_proj_abc = np.zeros((self.uv.shape[0], 2, 3))
        j_proj_abc[:, 0, 0] = 1.0
        j_proj_abc[:, 1, 1] = 1.0
        j_proj_abc[:, :, 2] = -self.uv
        j_up = j_normalize @ (j_proj_abc @ tangent_basis)  # (N, 2, 2)

        # Latitude Jacobian: d(sin lat)/d(abc) is the ray itself.
        j_lat = self.rays @ tangent_basis  # (N, 2)

        gradient = (up_weight[:, None] * np.einsum("nij,ni->nj", j_up, up_residual)).sum(axis=0)
        gradient += (lat_weight[:, None] * j_lat * lat_residual[:, None]).sum(axis=0)

        hessian = (up_weight[:, None, None] * np.einsum("nij,nik->njk", j_up, j_up)).sum(axis=0)
        hessian += (lat_weight[:, None, None] * j_lat[:, :, None] * j_lat[:, None, :]).sum(axis=0)
        return gradient, hessian


def fit_gravity(
    up_field: NDArray,
    up_confidence: NDArray,
    latitude_field: NDArray,
    latitude_confidence: NDArray,
    focal_x_px: float,
    focal_y_px: float,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> GravityFit:
    """Fit roll and pitch to the field-net outputs, focal fixed.

    Args:
        up_field: (2, h, w) or (1, 2, h, w) predicted up directions.
        up_confidence: (h, w) or (1, h, w) per-pixel confidence in [0, 1].
        latitude_field: (1, h, w) or (1, 1, h, w) predicted latitudes, radians.
        latitude_confidence: (h, w) or (1, h, w) per-pixel confidence.
        focal_x_px / focal_y_px: fixed focal in net-resolution pixels
            (original focal times the preprocessor's per-axis resize scale).
        num_steps: LM iteration budget (GeoCalib default 30, early stop active).
    """
    geometry = _PerspectiveGeometry(
        np.asarray(up_field).squeeze(),
        np.asarray(up_confidence).squeeze(),
        np.asarray(latitude_field).squeeze(),
        np.asarray(latitude_confidence).squeeze(),
        focal_x_px,
        focal_y_px,
    )

    vec = gravity_vec_from_roll_pitch(0.0, 0.0)
    lambda_damping = DEFAULT_INITIAL_LAMBDA
    *_, initial_cost = geometry.residuals_and_costs(vec)
    prev_cost = initial_cost
    stop_step = num_steps

    for step in range(num_steps):
        gradient, hessian = geometry.gradient_and_hessian(vec, spherical_j_plus(vec))

        damping = np.maximum(np.diag(hessian) * lambda_damping, 1e-6)
        try:
            delta = np.linalg.solve(hessian + np.diag(damping), gradient)
        except np.linalg.LinAlgError:
            delta = np.zeros(2)

        vec = spherical_plus(vec, delta)  # every step is accepted, no rollback
        *_, new_cost = geometry.residuals_and_costs(vec)

        lambda_damping = float(
            np.clip(
                lambda_damping * (10.0 if new_cost > prev_cost else 0.1),
                LAMBDA_MIN,
                LAMBDA_MAX,
            )
        )

        if abs(new_cost - prev_cost) <= STOP_ATOL + STOP_RTOL * abs(prev_cost):
            stop_step = min(step + 1, stop_step)
            break
        prev_cost = new_cost

    # Uncertainty: the Hessian in (roll, pitch) coordinates at the solution.
    _, hessian_rp = geometry.gradient_and_hessian(vec, j_roll_pitch(vec))
    covariance = np.linalg.inv(hessian_rp)
    eigenvalues = np.linalg.eigvalsh(covariance)

    roll, pitch = roll_pitch_from_gravity_vec(vec)
    *_, final_cost = geometry.residuals_and_costs(vec)
    return GravityFit(
        roll_rad=roll,
        pitch_rad=pitch,
        roll_uncertainty_rad=float(np.sqrt(max(covariance[0, 0], 0.0))),
        pitch_uncertainty_rad=float(np.sqrt(max(covariance[1, 1], 0.0))),
        gravity_uncertainty_rad=float(np.sqrt(max(eigenvalues[-1], 0.0))),
        initial_cost=float(initial_cost),
        final_cost=float(final_cost),
        stop_step=stop_step,
    )
