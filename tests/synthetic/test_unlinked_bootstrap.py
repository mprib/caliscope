"""Bootstrap method selection from the state of the image points."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import caliscope.core.bootstrap_pose.bootstrap_extrinsics as bootstrap_module
from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints
from caliscope.exceptions import CalibrationWarning
from caliscope.synthetic.scene_factories import default_ring_scene


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ImagePoints]]:
    """Replace each bootstrap variant with a recorder of which ran and what it received."""
    recorded: list[tuple[str, ImagePoints]] = []

    def recorder(name: str):
        def variant(points: ImagePoints, cameras: CameraArray) -> CameraArray:
            recorded.append((name, points))
            return cameras

        return variant

    for name in ("bootstrap_extrinsics_from_object_geometry", "bootstrap_extrinsics_from_linked_points"):
        monkeypatch.setattr(bootstrap_module, name, recorder(name))
    return recorded


def test_dispatch_follows_object_geometry_and_drops_unlinked_rows(calls: list[tuple[str, ImagePoints]]):
    scene = default_ring_scene()
    cameras = scene.intrinsics_only_cameras()
    with_geometry = scene.image_points_noisy.df
    without_geometry = with_geometry.copy()
    without_geometry[["obj_loc_x", "obj_loc_y", "obj_loc_z"]] = np.nan

    bootstrap_module.bootstrap_extrinsics(ImagePoints(with_geometry), cameras)
    bootstrap_module.bootstrap_extrinsics(ImagePoints(without_geometry), cameras)
    assert [name for name, _ in calls] == [
        "bootstrap_extrinsics_from_object_geometry",
        "bootstrap_extrinsics_from_linked_points",
    ]

    unlinked = with_geometry[with_geometry["cam_id"] == with_geometry["cam_id"].min()].copy()
    unlinked["object_id"] = -1
    mixed = ImagePoints(pd.concat([with_geometry, unlinked], ignore_index=True))

    with pytest.warns(CalibrationWarning, match=f"dropped {len(unlinked)} of {len(mixed.df)}"):
        bootstrap_module.bootstrap_extrinsics(mixed, cameras)

    name, received = calls[-1]
    assert name == "bootstrap_extrinsics_from_object_geometry"
    assert len(received.df) == len(with_geometry)
    assert (received.df["object_id"] >= 0).all()
