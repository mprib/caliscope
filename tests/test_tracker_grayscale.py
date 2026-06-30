from caliscope.packets import PixelFormat


def test_charuco_tracker_declares_gray():
    from caliscope.core.charuco import Charuco
    from caliscope.trackers.charuco_tracker import CharucoTracker

    charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=3.0)
    tracker = CharucoTracker(charuco)
    assert tracker.pixel_format == PixelFormat.GRAY


def test_aruco_tracker_declares_gray():
    from caliscope.trackers.aruco_tracker import ArucoTracker

    tracker = ArucoTracker()
    assert tracker.pixel_format == PixelFormat.GRAY


def test_chessboard_tracker_declares_gray():
    from caliscope.core.chessboard import Chessboard
    from caliscope.trackers.chessboard_tracker import ChessboardTracker

    chessboard = Chessboard(rows=6, columns=9)
    tracker = ChessboardTracker(chessboard)
    assert tracker.pixel_format == PixelFormat.GRAY
