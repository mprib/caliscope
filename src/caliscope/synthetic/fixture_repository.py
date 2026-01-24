"""Repository for synthetic scene fixture persistence."""

from dataclasses import dataclass
from pathlib import Path

import rtoml

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.persistence import (
    PersistenceError,
    load_camera_array,
    load_image_points_csv,
    load_world_points_csv,
    save_camera_array,
    save_image_points_csv,
    save_world_points_csv,
)
from caliscope.synthetic.synthetic_scene import SyntheticScene


@dataclass
class SyntheticFixture:
    """Loaded synthetic scene data ready for use in tests.

    Contains the domain objects that tests need, without the
    SyntheticScene computation machinery.
    """

    name: str
    camera_array: CameraArray
    world_points: WorldPoints
    image_points_noisy: ImagePoints
    pixel_noise_sigma: float
    random_seed: int


class SyntheticFixtureRepository:
    """Persistence gateway for synthetic scene fixtures.

    Directory structure:
        <base_path>/
            camera_array.toml
            world_points.csv
            image_points.csv
            metadata.toml
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._camera_path = base_path / "camera_array.toml"
        self._world_points_path = base_path / "world_points.csv"
        self._image_points_path = base_path / "image_points.csv"
        self._metadata_path = base_path / "metadata.toml"

    def save(self, scene: SyntheticScene, name: str) -> None:
        """Save synthetic scene as fixture."""
        self.base_path.mkdir(parents=True, exist_ok=True)

        save_camera_array(scene.camera_array, self._camera_path)
        save_world_points_csv(scene.world_points, self._world_points_path)
        save_image_points_csv(scene.image_points_noisy, self._image_points_path)

        metadata = {
            "name": name,
            "pixel_noise_sigma": scene.pixel_noise_sigma,
            "random_seed": scene.random_seed,
            "n_cameras": scene.n_cameras,
            "n_frames": scene.n_frames,
        }
        with open(self._metadata_path, "w") as f:
            rtoml.dump(metadata, f)

    def load(self) -> SyntheticFixture:
        """Load fixture from directory."""
        if not self._metadata_path.exists():
            raise PersistenceError(f"Fixture not found at {self.base_path}")

        camera_array = load_camera_array(self._camera_path)
        world_points = load_world_points_csv(self._world_points_path)
        image_points = load_image_points_csv(self._image_points_path)

        with open(self._metadata_path) as f:
            metadata = rtoml.load(f)

        return SyntheticFixture(
            name=metadata["name"],
            camera_array=camera_array,
            world_points=world_points,
            image_points_noisy=image_points,
            pixel_noise_sigma=metadata["pixel_noise_sigma"],
            random_seed=metadata["random_seed"],
        )

    def exists(self) -> bool:
        """Check if fixture exists at this path."""
        return self._metadata_path.exists()


# Convenience functions
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "synthetic"


def save_fixture(scene: SyntheticScene, name: str) -> Path:
    """Save scene as fixture with snake_case directory name."""
    fixture_dir = FIXTURES_DIR / name.lower().replace(" ", "_")
    repo = SyntheticFixtureRepository(fixture_dir)
    repo.save(scene, name)
    return fixture_dir


def load_fixture(name: str) -> SyntheticFixture:
    """Load fixture by name."""
    fixture_dir = FIXTURES_DIR / name.lower().replace(" ", "_")
    repo = SyntheticFixtureRepository(fixture_dir)
    return repo.load()
