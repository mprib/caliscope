import numpy as np
from scipy.spatial.transform import Rotation

from caliscope.core.alignment import estimate_similarity_transform


def generate_cube_points(size: float = 1.0, center: tuple = (0, 0, 0)) -> np.ndarray:
    """Generate 8 points of a cube centered at given location."""
    cx, cy, cz = center
    half = size / 2.0
    points = np.array(
        [
            [cx - half, cy - half, cz - half],
            [cx + half, cy - half, cz - half],
            [cx + half, cy + half, cz - half],
            [cx - half, cy + half, cz - half],
            [cx - half, cy - half, cz + half],
            [cx + half, cy - half, cz + half],
            [cx + half, cy + half, cz + half],
            [cx - half, cy + half, cz + half],
        ],
        dtype=np.float64,
    )
    return points


def generate_planar_points(grid_size: int = 4, spacing: float = 1.0) -> np.ndarray:
    """Generate NxN grid of points on z=0 plane (like a calibration board)."""
    x = np.linspace(0, (grid_size - 1) * spacing, grid_size)
    y = np.linspace(0, (grid_size - 1) * spacing, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points.astype(np.float64)


def test_perfect_similarity_transform():
    """Phase 1: Perfect data should recover exact transform."""
    # Generate source points (cube)
    source_points = generate_cube_points(size=1.0, center=(0, 0, 0))

    # Define true transform
    true_rotation = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
    true_translation = np.array([2.0, -1.5, 3.0], dtype=np.float64)
    true_scale = 2.5

    # Apply true transform to get target points
    target_points = true_scale * (true_rotation @ source_points.T).T + true_translation

    # Estimate transform
    estimated_transform = estimate_similarity_transform(source_points, target_points)

    # Extract components
    estimated_rotation = estimated_transform.rotation
    estimated_translation = estimated_transform.translation
    estimated_scale = estimated_transform.scale

    # Compute errors
    rot_error = np.linalg.norm(true_rotation - estimated_rotation)
    trans_error = np.linalg.norm(true_translation - estimated_translation)
    scale_error = abs(true_scale - estimated_scale)

    # Apply estimated transform and compute RMSE
    transformed_source = estimated_transform.apply(source_points)
    rmse = np.sqrt(np.mean(np.sum((transformed_source - target_points) ** 2, axis=1)))

    # Assertions
    assert rot_error < 1e-10, f"Rotation error too large: {rot_error}"
    assert trans_error < 1e-10, f"Translation error too large: {trans_error}"
    assert scale_error < 1e-10, f"Scale error too large: {scale_error}"
    assert rmse < 1e-10, f"RMSE too large: {rmse}"

    print("✓ Perfect similarity transform test passed")
    print(f"  Scale: estimated={estimated_scale:.10f}, true={true_scale:.10f}, error={scale_error:.2e}")
    print(f"  Rotation error: {rot_error:.2e}")
    print(f"  Translation error: {trans_error:.2e}")
    print(f"  RMSE: {rmse:.2e}")


def test_noisy_similarity_transform():
    """Phase 2: Noisy data should recover transform within statistical bounds."""
    # Use planar geometry (more realistic for calibration board)
    source_points = generate_planar_points(grid_size=4, spacing=0.1)

    # Define true transform
    true_rotation = Rotation.from_euler("z", 15, degrees=True).as_matrix()  # Rotation around board normal
    true_translation = np.array([0.5, -0.3, 1.2], dtype=np.float64)
    true_scale = 0.01  # 0.01 m per reconstruction unit

    # Apply true transform
    target_points = true_scale * (true_rotation @ source_points.T).T + true_translation

    # Add Gaussian noise (deterministic)
    np.random.seed(42)
    sigma = 0.001
    noise = np.random.normal(0, sigma, target_points.shape)
    noisy_target = target_points + noise

    # Estimate transform
    estimated_transform = estimate_similarity_transform(source_points, noisy_target)

    # Compute RMSE on original (noise-free) target
    transformed_source = estimated_transform.apply(source_points)
    rmse = np.sqrt(np.mean(np.sum((transformed_source - target_points) ** 2, axis=1)))

    # Assertions
    assert rmse < sigma * 2, f"RMSE {rmse} exceeds 2σ bound for sigma={sigma}"
    assert abs(estimated_transform.scale - true_scale) < sigma * 10, "Scale error too large"

    print("✓ Noisy similarity transform test passed")
    print(f"  Scale: estimated={estimated_transform.scale:.6f}, true={true_scale:.6f}")
    print(f"  RMSE: {rmse:.6f} (2σ bound: {2 * sigma:.6f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Similarity Transform Tests (Debug Mode)")
    print("=" * 60)

    try:
        test_perfect_similarity_transform()
        print()
    except AssertionError as e:
        print(f"✗ Perfect test FAILED: {e}")
        print()

    try:
        test_noisy_similarity_transform()
        print()
    except AssertionError as e:
        print(f"✗ Noisy test FAILED: {e}")
        print()

    print("=" * 60)
    print("Debug run complete")
    print("=" * 60)
