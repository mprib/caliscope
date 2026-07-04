"""Repository for synthetic scene fixture persistence."""

from dataclasses import dataclass
from pathlib import Path

import rtoml

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.constraints import ConstraintSet
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.persistence import PersistenceError
from caliscope.synthetic.synthetic_scene import SyntheticScene

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SyntheticFixture:
    """Loaded synthetic scene data ready for use in tests."""

    name: str
    camera_array: CameraArray
    world_points: WorldPoints
    image_points_noisy: ImagePoints
    pixel_noise_sigma: float
    random_seed: int
    constraints: ConstraintSet | None = None


class SyntheticFixtureRepository:
    """Persistence gateway for synthetic scene fixtures.

    Directory structure:
        <base_path>/
            metadata.toml
            camera_array.toml
            image_points.csv
            world_points.csv
            constraints.toml     (optional)
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._camera_path = base_path / "camera_array.toml"
        self._world_points_path = base_path / "world_points.csv"
        self._image_points_path = base_path / "image_points.csv"
        self._metadata_path = base_path / "metadata.toml"
        self._constraints_path = base_path / "constraints.toml"

    def save(
        self,
        scene: SyntheticScene,
        name: str,
        constraints: ConstraintSet | None = None,
    ) -> None:
        """Save synthetic scene as fixture."""
        self.base_path.mkdir(parents=True, exist_ok=True)

        scene.camera_array.to_toml(self._camera_path)
        scene.world_points.to_csv(self._world_points_path)
        scene.image_points_noisy.to_csv(self._image_points_path)

        if constraints is not None:
            constraints.to_toml(self._constraints_path)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "pixel_noise_sigma": scene.pixel_noise_sigma,
            "random_seed": scene.random_seed,
        }
        with open(self._metadata_path, "w") as f:
            rtoml.dump(metadata, f)

    def load(self) -> SyntheticFixture:
        """Load fixture from directory."""
        if not self._metadata_path.exists():
            raise PersistenceError(f"Fixture not found at {self.base_path}")

        with open(self._metadata_path) as f:
            metadata = rtoml.load(f)

        version = metadata.get("schema_version", 1)
        if version < SCHEMA_VERSION:
            raise PersistenceError(
                f"Fixture at {self.base_path} uses schema v{version}. "
                f"Current is v{SCHEMA_VERSION}. Regenerate with save_fixture()."
            )

        camera_array = CameraArray.from_toml(self._camera_path)
        world_points = WorldPoints.from_csv(self._world_points_path)
        image_points = ImagePoints.from_csv(self._image_points_path)

        constraints = None
        if self._constraints_path.exists():
            constraints = ConstraintSet.from_toml(self._constraints_path)

        return SyntheticFixture(
            name=metadata["name"],
            camera_array=camera_array,
            world_points=world_points,
            image_points_noisy=image_points,
            pixel_noise_sigma=metadata["pixel_noise_sigma"],
            random_seed=metadata["random_seed"],
            constraints=constraints,
        )

    def exists(self) -> bool:
        """Check if fixture exists at this path."""
        return self._metadata_path.exists()


# Convenience functions
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "synthetic"


def save_fixture(
    scene: SyntheticScene,
    name: str,
    constraints: ConstraintSet | None = None,
) -> Path:
    """Save scene as fixture with snake_case directory name."""
    fixture_dir = FIXTURES_DIR / name.lower().replace(" ", "_")
    repo = SyntheticFixtureRepository(fixture_dir)
    repo.save(scene, name, constraints=constraints)
    return fixture_dir


def load_fixture(name: str) -> SyntheticFixture:
    """Load fixture by name."""
    fixture_dir = FIXTURES_DIR / name.lower().replace(" ", "_")
    repo = SyntheticFixtureRepository(fixture_dir)
    return repo.load()
