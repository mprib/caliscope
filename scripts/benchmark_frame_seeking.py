"""
One-time benchmark: OpenCV vs PyAV frame seeking performance.

Validates that PyAV provides meaningful speedup for random frame access
before proceeding with keyframe-aware seeking implementation (issue #846).

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


def benchmark_opencv_random(path: str, frame_indices: list[int]) -> float:
    """Benchmark random frame seeking with OpenCV. Return seconds."""
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


def benchmark_pyav_random(path: str, frame_indices: list[int], fps: float) -> float:
    """Benchmark random frame seeking with PyAV. Return seconds."""
    container = av.open(path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base)

    start = time.perf_counter()
    for idx in frame_indices:
        # Seek to timestamp (in stream time_base units)
        target_pts = int(idx / fps / time_base)
        container.seek(target_pts, stream=stream)
        # Decode one frame
        for frame in container.decode(stream):
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

    # Generate random frame indices (same for both backends)
    random.seed(42)  # Reproducible
    random_indices = [random.randint(0, frame_count - 1) for _ in range(num_random)]

    # Run benchmarks
    print(f"Sequential read ({num_sequential} frames):")

    t_opencv_seq = benchmark_opencv_sequential(video_path, num_sequential)
    opencv_seq_fps = num_sequential / t_opencv_seq
    print(f"  OpenCV: {t_opencv_seq:.2f}s ({opencv_seq_fps:.0f} fps)")

    t_pyav_seq = benchmark_pyav_sequential(video_path, num_sequential)
    pyav_seq_fps = num_sequential / t_pyav_seq
    print(f"  PyAV:   {t_pyav_seq:.2f}s ({pyav_seq_fps:.0f} fps)")

    print()
    print(f"Random seek ({num_random} frames):")

    t_opencv_rand = benchmark_opencv_random(video_path, random_indices)
    opencv_rand_fps = num_random / t_opencv_rand
    print(f"  OpenCV: {t_opencv_rand:.2f}s ({opencv_rand_fps:.0f} fps)")

    t_pyav_rand = benchmark_pyav_random(video_path, random_indices, fps)
    pyav_rand_fps = num_random / t_pyav_rand
    print(f"  PyAV:   {t_pyav_rand:.2f}s ({pyav_rand_fps:.0f} fps)")

    print()
    print("Summary:")
    seq_ratio = t_opencv_seq / t_pyav_seq
    seq_result = "faster" if t_pyav_seq < t_opencv_seq else "slower"
    print(f"  Sequential: PyAV is {seq_ratio:.1f}x {seq_result}")

    rand_ratio = t_opencv_rand / t_pyav_rand
    rand_result = "faster" if t_pyav_rand < t_opencv_rand else "slower"
    print(f"  Random:     PyAV is {rand_ratio:.1f}x {rand_result}")

    # Decision
    print()
    if t_pyav_rand < t_opencv_rand:
        speedup = t_opencv_rand / t_pyav_rand
        print(f"RESULT: PyAV provides {speedup:.1f}x speedup for random seeking.")
        print("Recommendation: Proceed with issue #846 (keyframe-aware seeking).")
    else:
        print("RESULT: PyAV is NOT faster for random seeking on this system.")
        print("Recommendation: Investigate further before proceeding with #846.")


if __name__ == "__main__":
    main()
