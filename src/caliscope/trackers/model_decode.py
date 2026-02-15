"""Pure numpy decode functions for pose estimation model outputs.

Supports two output formats:
- SimCC (Simulated Coordinate Classification): 1D coordinate vectors
- Heatmap: 2D spatial probability maps

Both return (keypoints, confidence) as numpy arrays.
"""

import numpy as np


def decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SimCC vectors to keypoint coordinates and confidence.

    SimCC (Simulated Coordinate Classification) represents coordinates as 1D
    probability distributions. Coordinates are recovered via argmax and scaled
    by the split ratio. Confidence is the minimum of x and y axis probabilities.

    Args:
        simcc_x: X-axis probability vectors, shape (batch, K, W_x)
        simcc_y: Y-axis probability vectors, shape (batch, K, H_y)
        simcc_split_ratio: Scaling factor from model coordinates to pixels

    Returns:
        Tuple of (keypoints, confidence):
        - keypoints: (K, 2) float32 array of (x, y) coordinates in model input space
        - confidence: (K,) float32 array of per-keypoint confidence scores [0, 1]

    Raises:
        ValueError: If batch size is not 1 or shapes are incompatible
    """
    if simcc_x.shape[0] != 1 or simcc_y.shape[0] != 1:
        raise ValueError(f"Only batch_size=1 supported, got {simcc_x.shape[0]}")

    if simcc_x.shape[1] != simcc_y.shape[1]:
        raise ValueError(f"Keypoint count mismatch: simcc_x has {simcc_x.shape[1]}, simcc_y has {simcc_y.shape[1]}")

    # Remove batch dimension
    simcc_x = simcc_x[0]  # (K, W_x)
    simcc_y = simcc_y[0]  # (K, H_y)

    simcc_x.shape[0]

    # Find peaks in each 1D distribution
    x_indices = np.argmax(simcc_x, axis=1)  # (K,)
    y_indices = np.argmax(simcc_y, axis=1)  # (K,)

    # Get confidence values at peaks
    x_confidence = np.max(simcc_x, axis=1)  # (K,)
    y_confidence = np.max(simcc_y, axis=1)  # (K,)

    # Scale indices to coordinates
    x_coords = x_indices.astype(np.float32) / simcc_split_ratio
    y_coords = y_indices.astype(np.float32) / simcc_split_ratio

    # Stack into (K, 2) array
    keypoints = np.stack([x_coords, y_coords], axis=1)

    # Overall confidence is minimum of x and y
    confidence = np.minimum(x_confidence, y_confidence).astype(np.float32)

    return keypoints, confidence


def decode_heatmap(
    heatmaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode 2D heatmaps to keypoint coordinates with quadratic sub-pixel refinement.

    Finds the peak of each heatmap and applies Taylor expansion around the peak
    to estimate sub-pixel location. Refinement is clamped to +/-0.5 pixels and
    skipped for peaks on boundaries.

    Args:
        heatmaps: Heatmap tensor, shape (K, H, W) where K is number of keypoints

    Returns:
        Tuple of (keypoints, confidence):
        - keypoints: (K, 2) float32 array of (x, y) coordinates in heatmap space
        - confidence: (K,) float32 array of peak values [0, 1]
    """
    K, H, W = heatmaps.shape

    # Flatten spatial dimensions to find argmax
    flat_heatmaps = heatmaps.reshape(K, -1)
    max_indices = np.argmax(flat_heatmaps, axis=1)  # (K,)

    # Convert flat indices to 2D coordinates
    y_peaks = (max_indices // W).astype(np.float32)
    x_peaks = (max_indices % W).astype(np.float32)

    # Extract confidence (peak values)
    confidence = np.max(flat_heatmaps, axis=1).astype(np.float32)

    # Sub-pixel refinement via quadratic approximation (Taylor expansion)
    # Only refine interior peaks (not on boundaries)
    dx = np.zeros(K, dtype=np.float32)
    dy = np.zeros(K, dtype=np.float32)

    for k in range(K):
        x_int = int(x_peaks[k])
        y_int = int(y_peaks[k])

        # Skip boundary peaks
        if x_int == 0 or x_int == W - 1 or y_int == 0 or y_int == H - 1:
            continue

        hm = heatmaps[k]

        # X-axis refinement: fit parabola through (x-1, x, x+1)
        # Second derivative: f''(x) = f(x-1) - 2*f(x) + f(x+1)
        # First derivative: f'(x) ≈ (f(x+1) - f(x-1)) / 2
        # Offset: -f'(x) / f''(x)
        dxx = hm[y_int, x_int - 1] - 2 * hm[y_int, x_int] + hm[y_int, x_int + 1]
        if abs(dxx) > 1e-6:
            dx_val = (hm[y_int, x_int - 1] - hm[y_int, x_int + 1]) / (2 * dxx)
            dx[k] = np.clip(dx_val, -0.5, 0.5)

        # Y-axis refinement: fit parabola through (y-1, y, y+1)
        dyy = hm[y_int - 1, x_int] - 2 * hm[y_int, x_int] + hm[y_int + 1, x_int]
        if abs(dyy) > 1e-6:
            dy_val = (hm[y_int - 1, x_int] - hm[y_int + 1, x_int]) / (2 * dyy)
            dy[k] = np.clip(dy_val, -0.5, 0.5)

    # Apply refinement
    x_refined = x_peaks + dx
    y_refined = y_peaks + dy

    # Stack into (K, 2) array
    keypoints = np.stack([x_refined, y_refined], axis=1)

    return keypoints, confidence
