import numpy as np

from caliscope.packets import PixelFormat
from caliscope.recording.frame_source import FrameSource


def test_frame_source_gray_returns_2d_array(test_video_path):
    """Gray mode extracts Y plane: 2D array, same height/width as BGR."""
    cam_id = 0

    bgr_source = FrameSource.from_path(test_video_path, cam_id=cam_id)
    bgr_packet = bgr_source.next_frame()
    bgr_source.close()

    gray_source = FrameSource.from_path(test_video_path, cam_id=cam_id, pixel_format=PixelFormat.GRAY)
    gray_packet = gray_source.next_frame()
    gray_source.close()

    assert bgr_packet is not None
    assert gray_packet is not None

    assert gray_packet.frame.ndim == 2
    assert bgr_packet.frame.ndim == 3
    assert gray_packet.frame.shape == bgr_packet.frame.shape[:2]
    assert gray_packet.pixel_format == PixelFormat.GRAY
    assert bgr_packet.pixel_format == PixelFormat.BGR


def test_frame_source_gray_matches_opencv_conversion(test_video_path):
    """Y-plane extraction produces values close to cv2.cvtColor grayscale."""
    import cv2

    cam_id = 0

    bgr_source = FrameSource.from_path(test_video_path, cam_id=cam_id)
    bgr_packet = bgr_source.next_frame()
    bgr_source.close()

    gray_source = FrameSource.from_path(test_video_path, cam_id=cam_id, pixel_format=PixelFormat.GRAY)
    gray_packet = gray_source.next_frame()
    gray_source.close()

    assert bgr_packet is not None and gray_packet is not None

    cv2_gray = cv2.cvtColor(bgr_packet.frame, cv2.COLOR_BGR2GRAY)
    # The Y plane is the original luma from the encoder. The BGR path
    # goes through YUV→BGR→gray, accumulating rounding error at each
    # conversion step. Mean differences of ~10 are normal — the Y plane
    # is the more faithful grayscale, not the less faithful one.
    diff = np.abs(gray_packet.frame.astype(np.int16) - cv2_gray.astype(np.int16))
    assert diff.mean() < 15.0
