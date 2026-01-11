"""
One-time benchmark: OpenCV vs PyAV frame seeking performance.

Compares three seeking strategies:
1. OpenCV exact seek (CAP_PROP_POS_FRAMES) - decodes from keyframe to exact frame
2. PyAV keyframe seek - seeks to nearest keyframe only (fast preview)
3. PyAV exact seek - seeks to keyframe, then decodes forward to exact frame

This validates the two-tier seeking strategy for issue #846:
- During slider drag: keyframe preview (fast)
- On slider release: exact frame (accurate)

Usage:
    uv run python scripts/benchmark_frame_seeking.py [video_path]

If no video provided, uses a test session video.
"""

import random
import sys
import time
from pathlib import Path

import av
import cv2

# Default test video if none provided
DEFAULT_VIDEO = Path(__file__).parent.parent / "tests/sessions/prerecorded_calibration/calibration/intrinsic/port_0.mp4"


def get_video_info(path: str) -> tuple[int, float]:
    """Return (frame_count, fps) for video."""
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps


def benchmark_opencv_sequential(path: str, num_frames: int) -> float:
    """Benchmark sequential frame reading with OpenCV. Return seconds."""
    cap = cv2.VideoCapture(path)
    start = time.perf_counter()
    for _ in range(num_frames):
        success, _ = cap.read()
        if not success:
            break
    cap.release()
    return time.perf_counter() - start


def benchmark_opencv_exact(path: str, frame_indices: list[int]) -> float:
    """Benchmark exact frame seeking with OpenCV. Return seconds.

    Uses CAP_PROP_POS_FRAMES which internally seeks to keyframe
    and decodes forward to the exact requested frame.
    """
    cap = cv2.VideoCapture(path)
    start = time.perf_counter()
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap.read()
    cap.release()
    return time.perf_counter() - start


def benchmark_pyav_sequential(path: str, num_frames: int) -> float:
    """Benchmark sequential frame reading with PyAV. Return seconds."""
    container = av.open(path)
    stream = container.streams.video[0]
    start = time.perf_counter()
    count = 0
    for frame in container.decode(stream):
        count += 1
        if count >= num_frames:
            break
    container.close()
    return time.perf_counter() - start


def benchmark_pyav_keyframe(path: str, frame_indices: list[int], fps: float) -> float:
    """Benchmark keyframe-only seeking with PyAV. Return seconds.

    Seeks to nearest keyframe and returns first frame after seek.
    Fast but may not return the exact requested frame.
    Good for preview during slider drag.
    """
    container = av.open(path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base)

    start = time.perf_counter()
    for idx in frame_indices:
        # Seek to timestamp (lands on keyframe at or before target)
        target_pts = int(idx / fps / time_base)
        container.seek(target_pts, stream=stream)
        # Decode one frame (the keyframe)
        for frame in container.decode(stream):
            break
    container.close()
    return time.perf_counter() - start


def benchmark_pyav_exact(path: str, frame_indices: list[int], fps: float) -> float:
    """Benchmark exact frame seeking with PyAV. Return seconds.

    Seeks to keyframe before target, then decodes forward to exact frame.
    Slower but returns the exact requested frame.
    Good for final frame on slider release.
    """
    container = av.open(path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base)

    start = time.perf_counter()
    for target_idx in frame_indices:
        # Seek to timestamp (lands on keyframe at or before target)
        target_pts = int(target_idx / fps / time_base)
        container.seek(target_pts, stream=stream)

        # Decode frames until we reach or pass the target
        for frame in container.decode(stream):
            # frame.pts is in time_base units, convert to frame index
            frame_idx = int(frame.pts * time_base * fps)
            if frame_idx >= target_idx:
                break
    container.close()
    return time.perf_counter() - start


def main():
    # Get video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = str(DEFAULT_VIDEO)

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    print(f"Video: {video_path}")

    # Get video info
    frame_count, fps = get_video_info(video_path)
    print(f"Frames: {frame_count}, FPS: {fps:.1f}")
    print()

    # Parameters
    num_sequential = min(500, frame_count)
    num_random = 100

    # Generate random frame indices (same for all backends)
    random.seed(42)  # Reproducible
    random_indices = [random.randint(0, frame_count - 1) for _ in range(num_random)]

    # Sequential benchmarks
    print(f"Sequential read ({num_sequential} frames):")

    t_opencv_seq = benchmark_opencv_sequential(video_path, num_sequential)
    print(f"  OpenCV: {t_opencv_seq:.2f}s ({num_sequential / t_opencv_seq:.0f} fps)")

    t_pyav_seq = benchmark_pyav_sequential(video_path, num_sequential)
    print(f"  PyAV:   {t_pyav_seq:.2f}s ({num_sequential / t_pyav_seq:.0f} fps)")

    # Random seek benchmarks
    print()
    print(f"Random seek ({num_random} frames):")

    t_opencv = benchmark_opencv_exact(video_path, random_indices)
    print(f"  OpenCV (exact):       {t_opencv:.2f}s ({num_random / t_opencv:.0f} fps)")

    t_pyav_kf = benchmark_pyav_keyframe(video_path, random_indices, fps)
    print(f"  PyAV (keyframe only): {t_pyav_kf:.2f}s ({num_random / t_pyav_kf:.0f} fps)")

    t_pyav_exact = benchmark_pyav_exact(video_path, random_indices, fps)
    print(f"  PyAV (exact):         {t_pyav_exact:.2f}s ({num_random / t_pyav_exact:.0f} fps)")

    # Summary
    print()
    print("Summary:")
    print(f"  Sequential:     PyAV is {t_opencv_seq / t_pyav_seq:.1f}x vs OpenCV")
    print(f"  Keyframe seek:  PyAV is {t_opencv / t_pyav_kf:.1f}x vs OpenCV exact")
    print(f"  Exact seek:     PyAV is {t_opencv / t_pyav_exact:.1f}x vs OpenCV exact")

    # Decision
    print()
    print("Two-tier strategy assessment:")
    if t_pyav_kf < t_opencv:
        print(f"  Keyframe preview: {t_opencv / t_pyav_kf:.1f}x faster (good for drag)")
    else:
        print("  Keyframe preview: NOT faster than OpenCV")

    if t_pyav_exact < t_opencv:
        print(f"  Exact frame:      {t_opencv / t_pyav_exact:.1f}x faster (good for release)")
    elif t_pyav_exact <= t_opencv * 1.2:  # Within 20%
        print("  Exact frame:      Similar to OpenCV (acceptable)")
    else:
        print("  Exact frame:      SLOWER than OpenCV (concerning)")


if __name__ == "__main__":
    main()
