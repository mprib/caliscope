import ast
import json

import numpy as np
import pandas as pd
import pytest

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.point_data import WorldPoints
from caliscope.export.blender_scene import write_blender_scene
from caliscope.tracker import Segment, WireFrameView


def _posed_camera(cam_id: int) -> CameraData:
    camera = CameraData.from_intrinsics(cam_id=cam_id, size=(1280, 720), focal_length=900.0)
    camera.rotation = np.eye(3)
    camera.translation = np.array([0.5 * cam_id, 0.0, 2.0])
    return camera


def _world_points() -> WorldPoints:
    rows = []
    for sync_index in (0, 3, 6):
        for keypoint_id in (0, 1, 2):
            rows.append(
                {
                    "sync_index": sync_index,
                    "object_id": 0,
                    "keypoint_id": keypoint_id,
                    "x_coord": 0.1 * keypoint_id,
                    "y_coord": 0.2 * sync_index,
                    "z_coord": 1.0,
                    "frame_time": float(sync_index),
                }
            )
    return WorldPoints(pd.DataFrame(rows))


def test_writes_compilable_scene_script(tmp_path):
    cameras = CameraArray({1: _posed_camera(1), 2: _posed_camera(2)})
    wireframe = WireFrameView(
        segments=(Segment(name="seg", color="r", point_A="a", point_B="b"),),
        point_names={"a": 0, "b": 1},
    )
    script_path = write_blender_scene(
        cameras, _world_points(), tmp_path / "scene.py", wireframe=wireframe, run_blender=False
    )

    source = script_path.read_text()
    ast.parse(source)  # generated script must be valid Python

    payload_json = source.split('json.loads(r"""')[1].split('""")')[0]
    payload = json.loads(payload_json)
    assert payload["frames"] == [0, 3, 6]
    assert len(payload["cameras"]) == 2
    assert payload["edges"] == [[0, 1]]
    group_names = {group["name"] for group in payload["groups"]}
    assert group_names == {"r", "ungrouped"}


def test_rejects_unposed_cameras(tmp_path):
    cameras = CameraArray({1: CameraData.from_intrinsics(cam_id=1, size=(1280, 720), focal_length=900.0)})
    with pytest.raises(ValueError, match="posed"):
        write_blender_scene(cameras, _world_points(), tmp_path / "scene.py", run_blender=False)
