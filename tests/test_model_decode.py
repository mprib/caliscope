"""Tests for model output decode functions.

Verifies mathematical correctness of SimCC and heatmap decoding using
synthetic data (no ONNX dependency).
"""

import numpy as np
import pytest

from caliscope.trackers.model_decode import decode_heatmap, decode_simcc


def test_decode_simcc_recovers_known_peak():
    """Verify SimCC decode recovers the correct coordinate from a known peak."""
    # Create SimCC vectors with a single keypoint
    # Peak at index 75, split_ratio=2.0 -> expected coordinate 37.5
    K = 1  # Single keypoint
    W_x = 192
    H_y = 256

    # Build probability vectors with peak at index 75
    simcc_x = np.zeros((1, K, W_x), dtype=np.float32)
    simcc_y = np.zeros((1, K, H_y), dtype=np.float32)

    peak_x_index = 75
    peak_y_index = 75
    peak_value_x = 0.95
    peak_value_y = 0.88

    simcc_x[0, 0, peak_x_index] = peak_value_x
    simcc_y[0, 0, peak_y_index] = peak_value_y

    # Decode with split_ratio=2.0
    split_ratio = 2.0
    keypoints, confidence = decode_simcc(simcc_x, simcc_y, split_ratio)

    # Verify coordinate recovery
    expected_x = peak_x_index / split_ratio
    expected_y = peak_y_index / split_ratio

    assert keypoints.shape == (K, 2)
    assert confidence.shape == (K,)

    assert keypoints[0, 0] == pytest.approx(expected_x, abs=1e-5)
    assert keypoints[0, 1] == pytest.approx(expected_y, abs=1e-5)

    # Confidence should be minimum of x and y peak values
    expected_confidence = min(peak_value_x, peak_value_y)
    assert confidence[0] == pytest.approx(expected_confidence, abs=1e-5)


def test_decode_heatmap_subpixel_refinement():
    """Verify heatmap decode achieves sub-pixel accuracy via quadratic refinement."""
    # Generate Gaussian blob centered at (37.3, 52.7) on 64x64 grid
    # Quadratic refinement should recover the true center within ~0.1-0.2px

    H, W = 64, 64
    K = 1  # Single keypoint

    # True center (sub-pixel)
    true_x = 37.3
    true_y = 52.7

    # Create coordinate grids
    x_grid = np.arange(W, dtype=np.float32)
    y_grid = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Generate Gaussian blob
    sigma = 2.0
    heatmap = np.exp(-((xx - true_x) ** 2 + (yy - true_y) ** 2) / (2 * sigma**2))
    heatmap = heatmap.astype(np.float32)

    # Reshape to (K, H, W)
    heatmaps = heatmap[np.newaxis, :, :]

    # Decode
    keypoints, confidence = decode_heatmap(heatmaps)

    # Verify shape
    assert keypoints.shape == (K, 2)
    assert confidence.shape == (K,)

    # Verify sub-pixel recovery (quadratic refinement should get within 0.25px)
    recovered_x = keypoints[0, 0]
    recovered_y = keypoints[0, 1]

    error_x = abs(recovered_x - true_x)
    error_y = abs(recovered_y - true_y)

    assert error_x < 0.25, f"X error {error_x:.4f}px exceeds threshold"
    assert error_y < 0.25, f"Y error {error_y:.4f}px exceeds threshold"

    # Confidence reflects the sampled peak value (slightly below 1.0 because
    # the Gaussian center doesn't land exactly on an integer pixel)
    assert confidence[0] > 0.95


if __name__ == "__main__":
    """Debug harness for manual inspection of decode functions."""
    print("=" * 60)
    print("SimCC Decode Test")
    print("=" * 60)

    # SimCC setup
    K = 1
    W_x = 192
    H_y = 256
    simcc_x = np.zeros((1, K, W_x), dtype=np.float32)
    simcc_y = np.zeros((1, K, H_y), dtype=np.float32)

    peak_x_index = 75
    peak_y_index = 75
    peak_value_x = 0.95
    peak_value_y = 0.88

    simcc_x[0, 0, peak_x_index] = peak_value_x
    simcc_y[0, 0, peak_y_index] = peak_value_y

    split_ratio = 2.0
    keypoints, confidence = decode_simcc(simcc_x, simcc_y, split_ratio)

    expected_x = peak_x_index / split_ratio
    expected_y = peak_y_index / split_ratio
    expected_conf = min(peak_value_x, peak_value_y)

    print(f"Peak indices: x={peak_x_index}, y={peak_y_index}")
    print(f"Peak values: x={peak_value_x:.3f}, y={peak_value_y:.3f}")
    print(f"Split ratio: {split_ratio}")
    print(f"\nExpected coordinate: ({expected_x:.1f}, {expected_y:.1f})")
    print(f"Recovered coordinate: ({keypoints[0, 0]:.1f}, {keypoints[0, 1]:.1f})")
    print(f"\nExpected confidence: {expected_conf:.3f}")
    print(f"Recovered confidence: {confidence[0]:.3f}")

    print("\n" + "=" * 60)
    print("Heatmap Decode Test")
    print("=" * 60)

    # Heatmap setup
    H, W = 64, 64
    K = 1
    true_x = 37.3
    true_y = 52.7

    x_grid = np.arange(W, dtype=np.float32)
    y_grid = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x_grid, y_grid)

    sigma = 2.0
    heatmap = np.exp(-((xx - true_x) ** 2 + (yy - true_y) ** 2) / (2 * sigma**2))
    heatmap = heatmap.astype(np.float32)
    heatmaps = heatmap[np.newaxis, :, :]

    keypoints, confidence = decode_heatmap(heatmaps)

    recovered_x = keypoints[0, 0]
    recovered_y = keypoints[0, 1]
    error_x = abs(recovered_x - true_x)
    error_y = abs(recovered_y - true_y)

    print(f"True center: ({true_x:.1f}, {true_y:.1f})")
    print(f"Recovered center: ({recovered_x:.3f}, {recovered_y:.3f})")
    print(f"\nError: x={error_x:.4f}px, y={error_y:.4f}px")
    print(f"Confidence: {confidence[0]:.4f}")

    # Integer peak location (before refinement)
    flat_heatmap = heatmap.reshape(-1)
    max_idx = np.argmax(flat_heatmap)
    peak_y_int = max_idx // W
    peak_x_int = max_idx % W
    print(f"\nInteger peak: ({peak_x_int}, {peak_y_int})")
    print(f"Sub-pixel shift: dx={recovered_x - peak_x_int:.3f}, dy={recovered_y - peak_y_int:.3f}")

    print("\n" + "=" * 60)
    print("All tests passed when run via pytest")
    print("=" * 60)
