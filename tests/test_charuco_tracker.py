"""CharucoTracker mirror-path tests, including the decisive CV2 correspondence test.

Board-frame orientation (established empirically by these tests, and the reason the
sign below is +t): in the OpenCV charuco board frame (getChessboardCorners: x right,
y down within the print), a camera that sees the front face unmirrored looks along +Z.
The front face normal is therefore -Z, the substrate extends toward +Z, and the back
face sits at z = +thickness. (The spec's rev-4 draft assumed -t; this harness's
front-view control test falsified that.)

Physical model / mounting convention (to be documented in docs/calibration_targets.md):
the mirror print (`save_mirror_image`, a horizontal flip of the front print) is mounted
on the back of the substrate flipped about its VERTICAL axis, edges aligned with the
front sheet. Flipping the mirror print about the vertical axis undoes its horizontal
flip, so the content at back-face board point (x, y, +t) equals the front print's
content at (x, y): the pattern occupies the same board XY coordinates on both faces.
A camera behind the board therefore sees an x-mirrored image of the pattern — exactly
what the tracker's mirror path (flip -> detect -> un-flip x) exists to handle.

The decisive question (CV2, two-sided-charuco-thickness spec): after the un-flip, does
the corner the tracker labels id N sit directly behind front corner N at (x_N, y_N, +t),
or at the horizontal mirror partner (W - x_N, y_N, +t)? These tests render genuine
front-side and back-side viewpoints of the physical model through a synthetic pinhole
camera, run the real detector path, and assert reprojection agreement against ground
truth. The print-pixel to board-meters mapping is resolved empirically with the real
detector (not assumed), because the whole point is that this path is parity-sensitive.
"""

import cv2
import numpy as np
import pytest

from caliscope.core.charuco import Charuco
from caliscope.trackers.charuco_tracker import CharucoTracker

# Synthetic pinhole camera: 1280x960, f=1000px, no distortion.
IMG_SIZE = (1280, 960)
K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])

THICKNESS_M = 0.006  # 6mm foam core
CAM_DISTANCE_M = 0.6


def _make_charuco() -> Charuco:
    # 4x5 board, 5cm squares -> 20cm x 25cm, 12 interior corners.
    return Charuco.from_squares(columns=4, rows=5, square_size_cm=5.0)


def _print_mapping(charuco: Charuco):
    """Resolve the board-meters -> print-pixel mapping empirically.

    Runs the real detector on the raw fronto-parallel print and picks the
    axis orientation (4 flip combinations) that matches getChessboardCorners.
    Returns (board_img, board_w_m, board_h_m, to_px) where to_px maps board
    meters to print pixels. Asserting the fit is sub-pixel guarantees the
    rendering harness shares the detector's own convention rather than an
    assumed one.
    """
    board_img = charuco.board_img(pixmap_scale=1000)
    img_h, img_w = board_img.shape[:2]
    corners_m = np.asarray(charuco.board.getChessboardCorners())[:, :2]
    square_m = float(charuco.board.getSquareLength())
    board_w_m = charuco.columns * square_m
    board_h_m = charuco.rows * square_m

    tracker = CharucoTracker(charuco)
    ids, img_loc = tracker.find_corners_single_frame(board_img, mirror=False)
    assert len(ids) >= 6, "detector failed on the raw print — harness broken"

    best = None
    for x_flip in (False, True):
        for y_flip in (False, True):
            px = corners_m[ids, 0] / board_w_m * img_w
            py = corners_m[ids, 1] / board_h_m * img_h
            if x_flip:
                px = img_w - px
            if y_flip:
                py = img_h - py
            err = float(np.sqrt(np.mean((px - img_loc[:, 0]) ** 2 + (py - img_loc[:, 1]) ** 2)))
            if best is None or err < best[0]:
                best = (err, x_flip, y_flip)

    err, x_flip, y_flip = best
    assert err < 2.0, f"no axis orientation fits the print (best RMSE {err:.1f}px)"

    def to_px(x_m: float, y_m: float) -> tuple[float, float]:
        px = x_m / board_w_m * img_w
        py = y_m / board_h_m * img_h
        if x_flip:
            px = img_w - px
        if y_flip:
            py = img_h - py
        return (px, py)

    return board_img, board_w_m, board_h_m, to_px


def _project(points_m: np.ndarray, rotation: np.ndarray, cam_center: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = (-rotation @ cam_center).reshape(3, 1)
    projected, _ = cv2.projectPoints(points_m.astype(np.float64), rvec, tvec, K, None)
    return projected.reshape(-1, 2)


def _render_plane(
    board_img: np.ndarray,
    board_w_m: float,
    board_h_m: float,
    to_px,
    plane_z: float,
    rotation: np.ndarray,
    cam_center: np.ndarray,
) -> np.ndarray:
    """Render the printed plane as seen by the synthetic camera.

    The four physical extremes of the printed region (at plane_z) are projected
    into the camera; the homography from their print pixels warps the actual
    generated board image, so the detector runs on realistic imagery.
    """
    extremes_m = [(0.0, 0.0), (board_w_m, 0.0), (board_w_m, board_h_m), (0.0, board_h_m)]
    src = np.array([to_px(x, y) for x, y in extremes_m], dtype=np.float32)
    obj = np.array([[x, y, plane_z] for x, y in extremes_m])
    dst = _project(obj, rotation, cam_center).astype(np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(board_img, homography, IMG_SIZE, borderValue=255)


def _front_camera(board_w_m: float, board_h_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera facing the front face: on the -Z side, looking along +Z.

    This is the orientation from which the pattern reads unmirrored, i.e. the
    production front-camera pose family recovered by PnP."""
    rotation = np.eye(3)
    center = np.array([board_w_m / 2, board_h_m / 2, -CAM_DISTANCE_M])
    return rotation, center


def _back_camera(board_w_m: float, board_h_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera facing the back face: on the +Z side, looking along -Z."""
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    center = np.array([board_w_m / 2, board_h_m / 2, CAM_DISTANCE_M])
    return rotation, center


@pytest.fixture(scope="module")
def scene():
    charuco = _make_charuco()
    board_img, board_w_m, board_h_m, to_px = _print_mapping(charuco)
    return charuco, board_img, board_w_m, board_h_m, to_px


def test_front_view_reprojects_onto_front_corners(scene):
    """Harness control: a +Z view of the front face detects without the mirror
    path and reprojects onto (x_N, y_N, 0). If this fails, the rendering model
    is broken and the mirror test proves nothing."""
    charuco, board_img, board_w_m, board_h_m, to_px = scene
    rotation, center = _front_camera(board_w_m, board_h_m)
    frame = _render_plane(board_img, board_w_m, board_h_m, to_px, 0.0, rotation, center)

    tracker = CharucoTracker(charuco)
    packet = tracker._detect(frame, cam_id=0)

    assert len(packet.keypoint_id) >= 6
    assert tracker._last_mirrored[0] is False

    corners_m = np.asarray(charuco.board.getChessboardCorners())[:, :2]
    expected_3d = np.column_stack([corners_m[packet.keypoint_id], np.zeros(len(packet.keypoint_id))])
    expected_px = _project(expected_3d, rotation, center)
    rmse = float(np.sqrt(np.mean(np.sum((expected_px - packet.img_loc) ** 2, axis=1))))
    assert rmse < 2.0, f"front-view reprojection RMSE {rmse:.2f}px — harness broken"


def test_back_view_corners_sit_directly_behind(scene):
    """The decisive CV2 test: a genuine back-side viewpoint through the real
    tracker mirror path. The corner labeled id N must reproject onto
    (x_N, y_N, +t) — the directly-behind hypothesis — and must NOT fit the
    horizontal mirror-partner hypothesis (W - x_N, y_N, +t). If this fails,
    the two-sided thickness constraint geometry (obj_loc z-stamp AND the
    distance-t ties) must be recomputed before any of it is implemented."""
    charuco, board_img, board_w_m, board_h_m, to_px = scene
    rotation, center = _back_camera(board_w_m, board_h_m)
    frame = _render_plane(board_img, board_w_m, board_h_m, to_px, THICKNESS_M, rotation, center)

    tracker = CharucoTracker(charuco)
    packet = tracker._detect(frame, cam_id=0)

    assert len(packet.keypoint_id) >= 6
    assert tracker._last_mirrored[0] is True, "back view did not exercise the mirror path"

    corners_m = np.asarray(charuco.board.getChessboardCorners())[:, :2]
    detected = corners_m[packet.keypoint_id]

    behind_3d = np.column_stack([detected, np.full(len(detected), THICKNESS_M)])
    behind_px = _project(behind_3d, rotation, center)
    rmse_behind = float(np.sqrt(np.mean(np.sum((behind_px - packet.img_loc) ** 2, axis=1))))

    partner_3d = np.column_stack([board_w_m - detected[:, 0], detected[:, 1], np.full(len(detected), THICKNESS_M)])
    partner_px = _project(partner_3d, rotation, center)
    rmse_partner = float(np.sqrt(np.mean(np.sum((partner_px - packet.img_loc) ** 2, axis=1))))

    assert rmse_behind < 2.0, (
        f"directly-behind hypothesis rejected: RMSE {rmse_behind:.2f}px (mirror-partner RMSE {rmse_partner:.2f}px)"
    )
    assert rmse_partner > 20.0, f"test is not decisive: mirror-partner RMSE {rmse_partner:.2f}px is too close"


def test_zero_thickness_back_view_keeps_shared_identity(scene):
    """Regression guard on the historical collapse: at thickness 0 both faces
    share object_id 0 and z=0, so BA fuses them into the same world points."""
    charuco, board_img, board_w_m, board_h_m, to_px = scene
    rotation, center = _back_camera(board_w_m, board_h_m)
    frame = _render_plane(board_img, board_w_m, board_h_m, to_px, 0.0, rotation, center)

    tracker = CharucoTracker(charuco)
    packet = tracker._detect(frame, cam_id=0)

    assert len(packet.keypoint_id) >= 6
    assert tracker._last_mirrored[0] is True
    assert np.all(packet.object_id == 0)
    assert np.all(packet.obj_loc[:, 2] == 0.0)


def test_thick_board_back_view_gets_back_face_identity(scene):
    """Thickness > 0 + mirrored detection -> object_id=1, keypoint ids
    unchanged (0..n-1), obj_loc stamped z=+t — and that obj_loc reprojects
    onto the observed img_loc, which is the spec's self-consistency claim."""
    _, board_img, board_w_m, board_h_m, to_px = scene
    thick = _make_charuco()
    thick.thickness_cm = THICKNESS_M * 100
    rotation, center = _back_camera(board_w_m, board_h_m)
    frame = _render_plane(board_img, board_w_m, board_h_m, to_px, THICKNESS_M, rotation, center)

    tracker = CharucoTracker(thick)
    packet = tracker._detect(frame, cam_id=0)

    assert len(packet.keypoint_id) >= 6
    assert np.all(packet.object_id == 1)
    assert packet.keypoint_id.max() < (thick.columns - 1) * (thick.rows - 1)
    assert np.allclose(packet.obj_loc[:, 2], THICKNESS_M)

    reprojected = _project(packet.obj_loc, rotation, center)
    rmse = float(np.sqrt(np.mean(np.sum((reprojected - packet.img_loc) ** 2, axis=1))))
    assert rmse < 2.0, f"back-face obj_loc does not reproject onto img_loc (RMSE {rmse:.2f}px)"


def test_thick_board_front_view_unchanged(scene):
    """Thickness only affects mirrored detections: a front view of a thick
    board keeps object_id 0 and z=0."""
    _, board_img, board_w_m, board_h_m, to_px = scene
    thick = _make_charuco()
    thick.thickness_cm = THICKNESS_M * 100
    rotation, center = _front_camera(board_w_m, board_h_m)
    frame = _render_plane(board_img, board_w_m, board_h_m, to_px, 0.0, rotation, center)

    tracker = CharucoTracker(thick)
    packet = tracker._detect(frame, cam_id=0)

    assert len(packet.keypoint_id) >= 6
    assert np.all(packet.object_id == 0)
    assert np.all(packet.obj_loc[:, 2] == 0.0)


def test_mirror_hint_is_remembered_per_camera(scene):
    """The flip-hint cache: after a mirrored detection, the next frame for the
    same cam_id tries the mirrored orientation first; a different cam_id is
    unaffected."""
    charuco, board_img, board_w_m, board_h_m, to_px = scene
    back_rot, back_center = _back_camera(board_w_m, board_h_m)
    back_frame = _render_plane(board_img, board_w_m, board_h_m, to_px, 0.0, back_rot, back_center)

    tracker = CharucoTracker(charuco)
    tracker._detect(back_frame, cam_id=3)
    assert tracker._last_mirrored == {3: True}

    front_rot, front_center = _front_camera(board_w_m, board_h_m)
    front_frame = _render_plane(board_img, board_w_m, board_h_m, to_px, 0.0, front_rot, front_center)
    tracker._detect(front_frame, cam_id=7)
    assert tracker._last_mirrored == {3: True, 7: False}


if __name__ == "__main__":
    from pathlib import Path

    debug_dir = Path(__file__).parent / "tmp" / "charuco_mirror"
    debug_dir.mkdir(parents=True, exist_ok=True)

    charuco = _make_charuco()
    board_img, board_w_m, board_h_m, to_px = _print_mapping(charuco)

    front_rot, front_center = _front_camera(board_w_m, board_h_m)
    front = _render_plane(board_img, board_w_m, board_h_m, to_px, 0.0, front_rot, front_center)
    cv2.imwrite(str(debug_dir / "front_view.png"), front)

    back_rot, back_center = _back_camera(board_w_m, board_h_m)
    back = _render_plane(board_img, board_w_m, board_h_m, to_px, THICKNESS_M, back_rot, back_center)
    cv2.imwrite(str(debug_dir / "back_view.png"), back)

    tracker = CharucoTracker(charuco)
    packet = tracker._detect(back, cam_id=0)
    overlay = cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)
    corners_m = np.asarray(charuco.board.getChessboardCorners())[:, :2]
    detected = corners_m[packet.keypoint_id]
    behind_3d = np.column_stack([detected, np.full(len(detected), THICKNESS_M)])
    behind_px = _project(behind_3d, back_rot, back_center)
    for (ox, oy), (ex, ey), kid in zip(packet.img_loc, behind_px, packet.keypoint_id):
        cv2.circle(overlay, (int(ox), int(oy)), 6, (0, 0, 255), 2)  # observed: red
        cv2.circle(overlay, (int(ex), int(ey)), 3, (0, 255, 0), -1)  # expected: green
        cv2.putText(overlay, str(kid), (int(ox) + 8, int(oy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite(str(debug_dir / "back_view_correspondence.png"), overlay)
    print(f"debug images in {debug_dir}")
