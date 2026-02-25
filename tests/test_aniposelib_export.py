"""Test aniposelib-compatible TOML export for CameraArray."""

from pathlib import Path
import numpy as np
import rtoml
import pytest

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.persistence import save_camera_array_aniposelib


def create_test_camera(cam_id: int, with_extrinsics: bool = True) -> CameraData:
    """Create a test camera with intrinsic and optional extrinsic calibration.

    Args:
        cam_id: Camera identifier
        with_extrinsics: Whether to include rotation and translation

    Returns:
        CameraData instance
    """
    rotation = None
    translation = None

    if with_extrinsics:
        # Create different poses for different cameras
        angle = cam_id * np.pi / 6  # 30 degrees per camera
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        translation = np.array([cam_id * 0.5, 0.0, 1.0])

    return CameraData(
        cam_id=cam_id,
        size=(1280, 720),
        matrix=np.array([[900.0 + cam_id * 10, 0.0, 640.0], [0.0, 900.0 + cam_id * 10, 360.0], [0.0, 0.0, 1.0]]),
        distortions=np.array([-0.3, 0.1, 0.0, 0.0, 0.0]),
        rotation=rotation,
        translation=translation,
    )


def test_aniposelib_export_format(tmp_path: Path) -> None:
    """Verify aniposelib TOML format structure and content."""
    # Create camera array with 3 cameras
    cameras = {i: create_test_camera(i) for i in range(3)}
    camera_array = CameraArray(cameras)

    # Export to aniposelib format
    output_path = tmp_path / "camera_array_aniposelib.toml"
    save_camera_array_aniposelib(camera_array, output_path)

    # Read back and verify structure
    data = rtoml.load(output_path)

    # Check top-level section keys
    assert "cam_0" in data
    assert "cam_1" in data
    assert "cam_2" in data
    assert "metadata" in data

    # Verify each camera section has correct fields
    for cam_id in range(3):
        section = data[f"cam_{cam_id}"]

        # Required fields
        assert section["name"] == f"cam_{cam_id}"
        assert section["size"] == [1280, 720]
        assert isinstance(section["matrix"], list)
        assert isinstance(section["distortions"], list)
        assert isinstance(section["rotation"], list)
        assert isinstance(section["translation"], list)

        # Verify structure
        assert len(section["matrix"]) == 3
        assert len(section["matrix"][0]) == 3
        assert len(section["distortions"]) == 5
        assert len(section["rotation"]) == 3  # Rodrigues vector, not 3x3 matrix
        assert len(section["translation"]) == 3

        # No Caliscope-specific fields
        assert "cam_id" not in section
        assert "error" not in section
        assert "exposure" not in section
        assert "grid_count" not in section
        assert "ignore" not in section
        assert "rotation_count" not in section
        assert "fisheye" not in section

    # Verify metadata
    assert data["metadata"]["adjusted"] is False
    assert data["metadata"]["error"] == 0.0


def test_aniposelib_only_exports_posed_cameras(tmp_path: Path) -> None:
    """Verify that only cameras with extrinsics are exported."""
    # Create cameras with mixed calibration state
    cameras = {
        0: create_test_camera(0, with_extrinsics=True),
        1: create_test_camera(1, with_extrinsics=False),  # No extrinsics
        2: create_test_camera(2, with_extrinsics=True),
    }
    camera_array = CameraArray(cameras)

    # Export to aniposelib format
    output_path = tmp_path / "camera_array_aniposelib.toml"
    save_camera_array_aniposelib(camera_array, output_path)

    # Read back and verify only posed cameras are present
    data = rtoml.load(output_path)

    assert "cam_0" in data
    assert "cam_1" not in data  # Skipped due to missing extrinsics
    assert "cam_2" in data
    assert "metadata" in data


def test_aniposelib_rotation_is_rodrigues(tmp_path: Path) -> None:
    """Verify rotation is stored as 3-element Rodrigues vector, not 3x3 matrix."""
    cameras = {0: create_test_camera(0)}
    camera_array = CameraArray(cameras)

    output_path = tmp_path / "camera_array_aniposelib.toml"
    save_camera_array_aniposelib(camera_array, output_path)

    data = rtoml.load(output_path)
    rotation = data["cam_0"]["rotation"]

    # Should be a flat list of 3 elements (Rodrigues), not nested 3x3
    assert len(rotation) == 3
    assert isinstance(rotation[0], (int, float))
    assert not isinstance(rotation[0], list)


def test_aniposelib_matrix_values(tmp_path: Path) -> None:
    """Verify camera matrix values are correctly exported."""
    cameras = {0: create_test_camera(0)}
    camera_array = CameraArray(cameras)

    output_path = tmp_path / "camera_array_aniposelib.toml"
    save_camera_array_aniposelib(camera_array, output_path)

    data = rtoml.load(output_path)
    matrix = np.array(data["cam_0"]["matrix"])

    # Verify expected values
    assert matrix[0, 0] == pytest.approx(900.0)
    assert matrix[1, 1] == pytest.approx(900.0)
    assert matrix[0, 2] == pytest.approx(640.0)
    assert matrix[1, 2] == pytest.approx(360.0)


if __name__ == "__main__":
    """Debug harness for visual inspection of aniposelib export format."""
    from pathlib import Path

    # Create debug output directory
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Create test camera array
    cameras = {i: create_test_camera(i) for i in range(3)}
    camera_array = CameraArray(cameras)

    # Export to aniposelib format
    output_path = debug_dir / "camera_array_aniposelib.toml"
    save_camera_array_aniposelib(camera_array, output_path)

    print(f"Aniposelib TOML exported to: {output_path}")
    print("\nFile contents:")
    print(output_path.read_text())

    # Verify it can be loaded back
    data = rtoml.load(output_path)
    print("\n\nLoaded data structure:")
    for key, value in data.items():
        print(f"\n[{key}]")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k} = {v}")
