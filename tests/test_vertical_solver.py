"""Analytic-field tests for the numpy gravity solver.

The solver's forward model renders the up and latitude fields a tilted pinhole
camera produces. Feeding it perfect fields built from a known (roll, pitch) with
uniform confidence, it must recover that orientation to well under 0.1 deg. The
round-trip between the up vector and (roll, pitch) is also checked directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from caliscope.estimators.vertical_solver import (
    fit_gravity,
    gravity_vec_from_roll_pitch,
    roll_pitch_from_gravity_vec,
)


def _analytic_fields(
    roll: float, pitch: float, focal: float, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perfect up/latitude fields for a camera at (roll, pitch), uniform confidence.

    Replicates the solver's pixel-grid geometry independently: principal point
    at the image center, rays through the pinhole, up = image-plane projection
    of world up, latitude = arcsin(ray . up).
    """
    vec = gravity_vec_from_roll_pitch(roll, pitch)
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    xy = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)
    center = np.array([width / 2, height / 2])
    uv = (xy - center) / np.array([focal, focal])

    up_unnormalized = vec[:2][None, :] - vec[2] * uv
    up_normalized = up_unnormalized / np.linalg.norm(up_unnormalized, axis=-1, keepdims=True)

    rays = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    sin_lat = np.clip(rays @ vec, -1 + 1e-6, 1 - 1e-6)

    up_field = up_normalized.T.reshape(2, height, width)
    latitude_field = np.arcsin(sin_lat).reshape(height, width)
    confidence = np.ones((height, width), dtype=np.float64)
    return up_field, confidence, latitude_field, confidence


@pytest.mark.parametrize(
    "roll, pitch",
    [
        (0.0, 0.0),
        (0.10, 0.05),
        (-0.20, 0.15),
        (0.35, -0.25),
        (-0.45, -0.10),
    ],
)
def test_fit_gravity_recovers_known_orientation(roll: float, pitch: float) -> None:
    focal = 300.0
    up_field, up_conf, lat_field, lat_conf = _analytic_fields(roll, pitch, focal, 96, 128)

    fit = fit_gravity(up_field, up_conf, lat_field, lat_conf, focal, focal)

    recovered = gravity_vec_from_roll_pitch(fit.roll_rad, fit.pitch_rad)
    truth = gravity_vec_from_roll_pitch(roll, pitch)
    angle_deg = np.degrees(np.arccos(np.clip(np.dot(recovered, truth), -1.0, 1.0)))
    assert angle_deg < 0.1


def test_fit_gravity_reports_low_cost_and_finite_uncertainty() -> None:
    focal = 300.0
    up_field, up_conf, lat_field, lat_conf = _analytic_fields(0.1, -0.08, focal, 96, 128)

    fit = fit_gravity(up_field, up_conf, lat_field, lat_conf, focal, focal)

    # Perfect fields drive the cost far below its starting value and leave a
    # finite, positive Hessian uncertainty.
    assert fit.final_cost < fit.initial_cost
    assert fit.gravity_uncertainty_rad > 0.0
    assert np.isfinite(fit.gravity_uncertainty_rad)
    assert 1 <= fit.stop_step <= 30


@pytest.mark.parametrize(
    "roll, pitch",
    [
        (0.0, 0.0),
        (0.5, 0.4),
        (-0.7, -0.5),
        (0.6, 0.3),
        (-0.3, 0.7),
    ],
)
def test_roll_pitch_vector_round_trip(roll: float, pitch: float) -> None:
    # Angles stay within the near-upright envelope the estimator operates in.
    # The eps in the roll denominator amplifies near the arcsin knee (roll
    # approaching +/-90 deg), where the round-trip degrades to ~3e-4 rad.
    vec = gravity_vec_from_roll_pitch(roll, pitch)
    assert np.isclose(np.linalg.norm(vec), 1.0)

    recovered_roll, recovered_pitch = roll_pitch_from_gravity_vec(vec)
    # roll_pitch_from_gravity_vec carries GeoCalib's eps = 1e-4 in the roll
    # denominator (a deliberate port quirk), so the round-trip is only exact to
    # that scale, not to machine precision.
    assert np.isclose(recovered_roll, roll, atol=2e-4)
    assert np.isclose(recovered_pitch, pitch, atol=2e-4)


if __name__ == "__main__":
    focal = 300.0
    for roll, pitch in [(0.0, 0.0), (0.2, -0.15)]:
        fields = _analytic_fields(roll, pitch, focal, 96, 128)
        fit = fit_gravity(*fields, focal, focal)
        print(
            f"truth roll={np.degrees(roll):.3f} pitch={np.degrees(pitch):.3f} -> "
            f"fit roll={np.degrees(fit.roll_rad):.3f} pitch={np.degrees(fit.pitch_rad):.3f} "
            f"(stop_step={fit.stop_step}, final_cost={fit.final_cost:.3e})"
        )
