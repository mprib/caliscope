"""
Chessboard domain tests: dataclass, persistence, and tracker.

Design decision: chessboard is used for intrinsic calibration only.
Extrinsic calibration uses ArUco (inherently unambiguous orientation).
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pytest

from caliscope.core.chessboard import Chessboard
from caliscope.persistence import PersistenceError
from caliscope.trackers.chessboard_tracker import ChessboardTracker, _subpix_window_half_width


def _row_major_grid(columns: int, rows: int, pitch_px: float) -> np.ndarray:
    """Synthetic raw findChessboardCorners output: (N, 1, 2), row-major."""
    xs, ys = np.meshgrid(np.arange(columns) * pitch_px, np.arange(rows) * pitch_px)
    return np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2).astype(np.float32)


def test_subpix_window_small_board_shrinks():
    """A 16 px-pitch board (OpenCap scale) gets a 4 px half-window, not 11.

    Measured on OpenCap Cam0: the fixed 11 px window inflated the homography
    residual to 4-8 px; floor(16 / 4) = 4 px keeps it near 0.1 px.
    """
    corners = _row_major_grid(columns=9, rows=6, pitch_px=16.0)
    assert _subpix_window_half_width(corners, columns=9, rows=6) == 4


def test_subpix_window_large_board_hits_ceiling():
    """GUI-scale board (60 px pitch) clamps to the old 11 px ceiling — unchanged."""
    corners = _row_major_grid(columns=9, rows=6, pitch_px=60.0)
    assert _subpix_window_half_width(corners, columns=9, rows=6) == 11


def test_subpix_window_uses_min_over_both_axes():
    """Vertical foreshortening drives the window: a board wide in X but squashed
    in Y (12 px vertical pitch) must pick the vertical minimum, not the 60 px
    horizontal pitch.
    """
    xs, ys = np.meshgrid(np.arange(9) * 60.0, np.arange(6) * 12.0)
    corners = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2).astype(np.float32)
    assert _subpix_window_half_width(corners, columns=9, rows=6) == 3  # floor(12 / 4)


# ── Dataclass ────────────────────────────────────────────────────────────────


def test_chessboard_immutable():
    """Frozen dataclass should reject mutation."""
    cb = Chessboard(rows=6, columns=9)
    with pytest.raises(FrozenInstanceError):
        cb.rows = 5  # type: ignore[misc]


def test_get_object_points_shape():
    """Object points should have correct shape."""
    cb = Chessboard(rows=6, columns=9)
    points = cb.get_object_points()
    assert points.shape == (54, 3)
    assert points.dtype == np.float32


def test_get_object_points_origin():
    """First point should be at origin."""
    cb = Chessboard(rows=6, columns=9)
    points = cb.get_object_points()
    np.testing.assert_array_equal(points[0], [0, 0, 0])


def test_get_object_points_spacing():
    """Adjacent points should be unit spacing apart."""
    cb = Chessboard(rows=6, columns=9)
    points = cb.get_object_points()
    np.testing.assert_array_almost_equal(points[1], [1.0, 0, 0])
    np.testing.assert_array_almost_equal(points[9], [0, 1.0, 0])


def test_get_object_points_planar():
    """All points should have Z=0 (planar board)."""
    cb = Chessboard(rows=6, columns=9)
    points = cb.get_object_points()
    np.testing.assert_array_equal(points[:, 2], np.zeros(54))


# ── Persistence ──────────────────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path):
    """Save/load round-trip should produce identical Chessboard."""
    original = Chessboard(rows=6, columns=9)
    file_path = tmp_path / "chessboard.toml"
    original.to_toml(file_path)
    loaded = Chessboard.from_toml(file_path)

    assert loaded.rows == original.rows
    assert loaded.columns == original.columns


def test_load_nonexistent_file(tmp_path: Path):
    """Load from nonexistent path should raise PersistenceError."""
    with pytest.raises(PersistenceError, match="not found"):
        Chessboard.from_toml(tmp_path / "nonexistent.toml")


def test_load_legacy_toml_with_square_size_cm(tmp_path: Path):
    """Loading TOML with legacy square_size_cm field should succeed."""
    file_path = tmp_path / "legacy_chessboard.toml"
    file_path.write_text("rows = 6\ncolumns = 9\nsquare_size_cm = 2.5\n")

    loaded = Chessboard.from_toml(file_path)

    assert loaded.rows == 6
    assert loaded.columns == 9


def test_save_creates_parent_directory(tmp_path: Path):
    """Save should create parent directories if they don't exist."""
    original = Chessboard(rows=6, columns=9)
    file_path = tmp_path / "nested" / "path" / "chessboard.toml"
    original.to_toml(file_path)

    assert file_path.exists()
    assert Chessboard.from_toml(file_path).rows == original.rows


# ── Tracker (real frames) ───────────────────────────────────────────────────

TEST_DATA_DIR = Path("tests/sessions/chessboard_intrinsic")


@pytest.fixture
def tracker() -> ChessboardTracker:
    return ChessboardTracker(Chessboard(rows=6, columns=9))


def test_detection_finds_all_corners(tracker: ChessboardTracker) -> None:
    """Verify detection returns all 54 corners with correct shapes."""
    frame_path = TEST_DATA_DIR / "cam_0_frame_100.jpg"
    if not frame_path.exists():
        pytest.skip(f"Test data not extracted: {frame_path}")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    packet = tracker.get_points(frame)

    assert len(packet.keypoint_id) == 54, "Should detect all 54 corners (6*9)"
    assert packet.img_loc.shape == (54, 2)
    assert packet.obj_loc.shape == (54, 3)


def test_no_board_returns_empty_packet(tracker: ChessboardTracker) -> None:
    """Verify empty PointPacket when no board is visible."""
    frame_path = TEST_DATA_DIR / "cam_0_frame_000.jpg"
    if not frame_path.exists():
        pytest.skip(f"Test data not extracted: {frame_path}")

    frame = cv2.imread(str(frame_path))
    assert frame is not None
    packet = tracker.get_points(frame)

    assert len(packet.keypoint_id) == 0
    assert packet.img_loc.shape == (0, 2)
    assert packet.obj_loc.shape == (0, 3)


def test_cross_camera_consistency(tracker: ChessboardTracker) -> None:
    """
    Multiple cameras agree on corner ordering for the same board position.

    Relies on images being right-side up (user-configured rotation_count).
    """
    frame_paths = [TEST_DATA_DIR / f"cam_{i}_frame_1070.jpg" for i in range(4)]

    available_frames = [(p, cv2.imread(str(p))) for p in frame_paths if p.exists()]
    if len(available_frames) < 2:
        pytest.skip("Need at least 2 camera frames for cross-camera test")

    packets = []
    for path, frame in available_frames:
        if frame is None:
            continue
        packet = tracker.get_points(frame)
        if len(packet.keypoint_id) > 0:
            packets.append((path, packet))

    if len(packets) < 2:
        pytest.skip("Need at least 2 successful detections for comparison")

    reference_ids = packets[0][1].keypoint_id
    for path, packet in packets[1:]:
        np.testing.assert_array_equal(
            packet.keypoint_id,
            reference_ids,
            err_msg=f"Corner ordering mismatch: {packets[0][0]} vs {path}",
        )


if __name__ == "__main__":
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    chessboard = Chessboard(rows=6, columns=9)
    tracker_ = ChessboardTracker(chessboard)

    frame_path = TEST_DATA_DIR / "cam_0_frame_100.jpg"
    if not frame_path.exists():
        print(f"Test data not found at {frame_path}")
        print("Extract test PNGs first (see milestone spec).")
        exit(1)

    frame = cv2.imread(str(frame_path))
    packet = tracker_.get_points(frame)

    annotated = frame.copy()
    for pid, loc in zip(packet.keypoint_id, packet.img_loc):
        x, y = int(loc[0]), int(loc[1])
        cv2.circle(annotated, (x, y), 5, (0, 220, 0), 2)
        cv2.putText(annotated, str(pid), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 220, 0), 1)

    cv2.imwrite(str(debug_dir / "chessboard_detection.jpg"), annotated)
    print(f"Detected {len(packet.keypoint_id)} corners, saved to {debug_dir}")
