"""D5: Fixture repository round-trip tests.

Tests save/load for single-object and multi-object (aruco) scenes,
with and without constraints.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from caliscope.core.aruco_marker import ArucoMarker, ArucoMarkerSet
from caliscope.persistence import PersistenceError
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.fixture_repository import (
    SyntheticFixtureRepository,
)
from caliscope.synthetic.scene_factories import aruco_scene, default_ring_scene
from caliscope.synthetic.se3_pose import SE3Pose
from caliscope.synthetic.trajectory import Trajectory


class TestSingleObjectRoundTrip:
    def test_camera_array_round_trips(self, tmp_path) -> None:
        scene = default_ring_scene()
        repo = SyntheticFixtureRepository(tmp_path / "fix")
        repo.save(scene, "test")
        fixture = repo.load()

        for cam_id in scene.camera_array.cameras:
            orig = scene.camera_array.cameras[cam_id]
            loaded = fixture.camera_array.cameras[cam_id]
            np.testing.assert_allclose(orig.matrix, loaded.matrix, atol=1e-6)
            np.testing.assert_allclose(orig.distortions, loaded.distortions, atol=1e-6)

    def test_image_points_round_trip(self, tmp_path) -> None:
        scene = default_ring_scene()
        repo = SyntheticFixtureRepository(tmp_path / "fix")
        repo.save(scene, "test")
        fixture = repo.load()

        orig = scene.image_points_noisy.df.sort_values(
            ["sync_index", "cam_id", "object_id", "keypoint_id"]
        ).reset_index(drop=True)
        loaded = fixture.image_points_noisy.df.sort_values(
            ["sync_index", "cam_id", "object_id", "keypoint_id"]
        ).reset_index(drop=True)

        np.testing.assert_allclose(
            orig[["img_loc_x", "img_loc_y"]].to_numpy(),
            loaded[["img_loc_x", "img_loc_y"]].to_numpy(),
            atol=0.01,
        )

    def test_metadata_round_trips(self, tmp_path) -> None:
        scene = default_ring_scene(pixel_noise_sigma=0.3, random_seed=99)
        repo = SyntheticFixtureRepository(tmp_path / "fix")
        repo.save(scene, "my_scene")
        fixture = repo.load()

        assert fixture.name == "my_scene"
        assert fixture.pixel_noise_sigma == pytest.approx(0.3)
        assert fixture.random_seed == 99

    def test_no_constraints_loads_as_none(self, tmp_path) -> None:
        scene = default_ring_scene()
        repo = SyntheticFixtureRepository(tmp_path / "fix")
        repo.save(scene, "test")
        fixture = repo.load()
        assert fixture.constraints is None


class TestArucoWithConstraints:
    def _make_aruco_scene(self):
        camera_array = CameraSynthesizer().add_ring(n=4, radius=2.0, height=0.5).build()
        markers = {
            0: ArucoMarker(0, 0.1),
            1: ArucoMarker(1, 0.1, static=True),
        }
        marker_set = ArucoMarkerSet(dictionary=cv2.aruco.DICT_4X4_50, markers=markers)
        trajectories = {
            0: Trajectory.orbital(n_frames=20, radius=0.5),
            1: Trajectory.stationary(
                n_frames=20,
                pose=SE3Pose.from_axis_angle(
                    axis=np.array([0.0, 0.0, 1.0]),
                    angle_rad=0.0,
                    translation=np.array([0.3, -0.2, 0.0]),
                ),
            ),
        }
        return aruco_scene(
            marker_set=marker_set,
            trajectories=trajectories,
            camera_array=camera_array,
        )

    def test_constraints_round_trip(self, tmp_path) -> None:
        scene, constraints = self._make_aruco_scene()
        repo = SyntheticFixtureRepository(tmp_path / "fix")
        repo.save(scene, "aruco", constraints=constraints)
        fixture = repo.load()

        assert fixture.constraints is not None
        assert len(fixture.constraints.distances) == len(constraints.distances)
        assert fixture.constraints.static_object_ids == constraints.static_object_ids

        for orig, loaded in zip(constraints.distances, fixture.constraints.distances):
            assert orig.object_id_a == loaded.object_id_a
            assert orig.distance == pytest.approx(loaded.distance)


class TestSchemaVersion:
    def test_v1_raises_persistence_error(self, tmp_path) -> None:
        scene = default_ring_scene()
        repo = SyntheticFixtureRepository(tmp_path / "fix")
        repo.save(scene, "test")

        # Downgrade schema version in metadata
        import rtoml

        meta_path = tmp_path / "fix" / "metadata.toml"
        with open(meta_path) as f:
            data = rtoml.load(f)
        data["schema_version"] = 1
        with open(meta_path, "w") as f:
            rtoml.dump(data, f)

        with pytest.raises(PersistenceError, match="v1"):
            repo.load()

    def test_missing_fixture_raises(self, tmp_path) -> None:
        repo = SyntheticFixtureRepository(tmp_path / "nonexistent")
        with pytest.raises(PersistenceError):
            repo.load()


if __name__ == "__main__":
    from pathlib import Path
    import tempfile

    debug_dir = Path(__file__).parent.parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print("Testing single-object round-trip...")
        t = TestSingleObjectRoundTrip()
        t.test_camera_array_round_trips(tmp / "t1")
        t.test_image_points_round_trip(tmp / "t2")
        t.test_metadata_round_trips(tmp / "t3")
        t.test_no_constraints_loads_as_none(tmp / "t4")
        print("  PASSED")

        print("Testing aruco with constraints...")
        t2 = TestArucoWithConstraints()
        t2.test_constraints_round_trip(tmp / "t5")
        print("  PASSED")

        print("Testing schema version...")
        t3 = TestSchemaVersion()
        t3.test_v1_raises_persistence_error(tmp / "t6")
        t3.test_missing_fixture_raises(tmp / "t7")
        print("  PASSED")
