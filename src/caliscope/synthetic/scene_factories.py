"""Factory functions for common synthetic calibration scenes."""

from caliscope.synthetic.calibration_object import CalibrationObject
from caliscope.synthetic.camera_synthesizer import CameraSynthesizer
from caliscope.synthetic.synthetic_scene import SyntheticScene
from caliscope.synthetic.trajectory import Trajectory


def default_ring_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """Standard 4-camera ring with full orbital trajectory.

    Configuration:
    - 4 cameras in ring, radius=2000mm, height=500mm
    - 5x7 planar grid, spacing=50mm
    - 20 frames, orbital radius=200mm, full 360 degree orbit
    """
    camera_array = CameraSynthesizer().add_ring(n=4, radius_mm=2000.0, height_mm=500.0).build()

    calibration_object = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)

    trajectory = Trajectory.orbital(
        n_frames=20,
        radius_mm=200.0,
        arc_extent_deg=360.0,
        tumble_rate=1.0,
    )

    return SyntheticScene(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def sparse_coverage_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """4 cameras, partial arc (cameras don't all see same frames).

    Tests scenarios where cameras have limited shared visibility.
    """
    camera_array = CameraSynthesizer().add_ring(n=4, radius_mm=2000.0, height_mm=500.0).build()

    calibration_object = CalibrationObject.planar_grid(rows=5, cols=7, spacing_mm=50.0)

    trajectory = Trajectory.orbital(
        n_frames=20,
        radius_mm=400.0,  # Larger radius = less overlap
        arc_extent_deg=180.0,  # Half orbit
        tumble_rate=0.5,
    )

    return SyntheticScene(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )


def quick_test_scene(
    pixel_noise_sigma: float = 0.5,
    random_seed: int = 42,
) -> SyntheticScene:
    """Minimal scene for fast tests (5 frames, small grid)."""
    camera_array = CameraSynthesizer().add_ring(n=4, radius_mm=2000.0, height_mm=500.0).build()

    calibration_object = CalibrationObject.planar_grid(rows=3, cols=4, spacing_mm=50.0)

    trajectory = Trajectory.orbital(
        n_frames=5,
        radius_mm=200.0,
        arc_extent_deg=180.0,
        tumble_rate=0.5,
    )

    return SyntheticScene(
        camera_array=camera_array,
        calibration_object=calibration_object,
        trajectory=trajectory,
        pixel_noise_sigma=pixel_noise_sigma,
        random_seed=random_seed,
    )
