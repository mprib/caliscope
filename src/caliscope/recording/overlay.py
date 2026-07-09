"""Draw tracked-point overlays onto decoded frames.

Shared by the reconstruction overlay-video sink (OverlayVideoWriter) and any other
caller that needs points drawn on a frame. Pure CV: takes an ndarray, returns an
ndarray, no I/O.
"""

from typing import Any, Callable

import cv2
from numpy.typing import NDArray

from caliscope.packets import PixelFormat, PointPacket


def draw_scatter_overlay(
    frame: NDArray[Any],
    points: PointPacket | None,
    draw_instructions: Callable[[int], dict] | None,
    pixel_format: PixelFormat,
) -> NDArray[Any]:
    """Return a 3-channel BGR copy of `frame` with each tracked point drawn as a circle.

    A GRAY frame is converted to BGR unconditionally, so the result is always a valid
    HxWx3 array safe for a bgr24 encoder even when `points` is None and nothing is drawn.
    `draw_instructions(keypoint_id)` supplies each circle's radius, color, and thickness.
    """
    if pixel_format == PixelFormat.GRAY:
        drawn = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        drawn = frame.copy()

    if points is not None and draw_instructions is not None:
        for keypoint_id, coord in zip(points.keypoint_id, points.img_loc):
            x = round(coord[0])
            y = round(coord[1])
            params = draw_instructions(keypoint_id)
            cv2.circle(drawn, (x, y), params["radius"], params["color"], params["thickness"])

    return drawn
