"""Charuco board domain + preview tests.

Focus: the two dense-board preview crashes fixed for issue #978.
  1. board_img must not abort on the quasi-periodic generateImage failure sizes.
  2. The dictionary pool must auto-fit the board so marker ids never overflow.

The pure-domain tests need no Qt. The single rendering test constructs a
QApplication and converts to a QPixmap, which is screen-backed; see the
offscreen platform note below.
"""

import os
from pathlib import Path

import pytest

from caliscope.core.charuco import (
    Charuco,
    DictionaryCapacityError,
    fit_dictionary_pool,
)
from caliscope.repositories.calibration_targets_repository import CalibrationTargetsRepository

# A QPixmap needs a platform plugin that can back it. Headless CI runners
# (notably macOS/Windows, which have no window server) return a null pixmap
# under the native plugin. The offscreen plugin has a raster backend that works
# headlessly everywhere, so force it before any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# Reporter board from issue #978: dense 28x17 ChArUco, DICT_4X4_250, 8.5x7 in.
def _reporter_board() -> Charuco:
    return Charuco(
        columns=28,
        rows=17,
        board_height=7,
        board_width=8.5,
        dictionary="DICT_4X4_250",
        units="inch",
        aruco_scale=0.75,
    )


# -- Fix 1: board_img render-with-retry --------------------------------------


def test_board_img_succeeds_across_failure_band():
    """The #978 repro: the reporter's dense board must render at every scale.

    generateImage aborts on a scattered set of output sizes (an integer-pixel
    rounding artifact), so the retry must absorb them. 200 is the preview scale;
    243 and 467 are confirmed lattice failures; 1000/2000 bracket the range.
    Every scale must yield an image, not raise cv2.error.
    """
    board = _reporter_board()
    for scale in (200, 243, 467, 1000, 2000):
        img = board.board_img(pixmap_scale=scale)
        assert img.size > 0


# -- Fix 2: fit_dictionary_pool ----------------------------------------------


def test_fit_dictionary_pool_widens_to_smallest_fitting_pool():
    """238 markers don't fit DICT_4X4_50/100 but fit DICT_4X4_250."""
    assert fit_dictionary_pool("DICT_4X4_50", 238) == "DICT_4X4_250"


def test_fit_dictionary_pool_shrinks_oversized_pool():
    """An oversized pool narrows to the smallest that still fits (always-smallest-fit).

    Nested pools make the shrink safe: the markers a printed board uses are
    unchanged. 40 markers fit DICT_4X4_50, so DICT_4X4_250 narrows to it.
    """
    assert fit_dictionary_pool("DICT_4X4_250", 40) == "DICT_4X4_50"


def test_fit_dictionary_pool_leaves_minimal_pool_unchanged():
    """A pool already at the smallest fit is returned unchanged."""
    assert fit_dictionary_pool("DICT_4X4_50", 40) == "DICT_4X4_50"


def test_fit_dictionary_pool_passes_through_non_laddered_dict():
    """Non-laddered families have no pool ladder and are returned unchanged."""
    assert fit_dictionary_pool("DICT_ARUCO_ORIGINAL", 100) == "DICT_ARUCO_ORIGINAL"


def test_fit_dictionary_pool_raises_over_family_max():
    """A board needing more than the family's largest pool can't be fit."""
    with pytest.raises(DictionaryCapacityError):
        fit_dictionary_pool("DICT_4X4_50", 1500)


def test_dictionary_capacity_error_names_needed_and_capacity():
    """The message must name both the shortfall and the real pool size."""
    # DICT_APRILTAG_16h5 holds 30 markers (fewer than 50, per the apriltag family).
    with pytest.raises(DictionaryCapacityError) as exc:
        fit_dictionary_pool("DICT_APRILTAG_16h5", 100)
    assert exc.value.needed == 100
    assert exc.value.capacity == 30
    assert "100" in str(exc.value)
    assert "30" in str(exc.value)


# -- Fix 2: normalization at the persistence boundary ------------------------


def test_to_toml_writes_fitted_dictionary_and_survives_reload(tmp_path: Path):
    """Saving a board that crosses a pool boundary writes the corrected dict."""
    path = tmp_path / "charuco.toml"
    board = _reporter_board()  # 238 markers, declared DICT_4X4_250
    # Force an undersized starting dictionary; saving must widen it to fit.
    board.dictionary = "DICT_4X4_50"
    board.to_toml(path)

    reloaded = Charuco.from_toml(path)
    assert reloaded.dictionary == "DICT_4X4_250"


def test_from_toml_corrects_hand_edited_undersized_dictionary(tmp_path: Path):
    """A hand-edited TOML with a mismatched dict is corrected on load (#978 startup)."""
    path = tmp_path / "intrinsic_charuco.toml"
    # Write a TOML directly (bypassing to_toml's normalization) with dims that
    # need 238 markers but an undersized DICT_4X4_50.
    path.write_text(
        "\n".join(
            [
                "columns = 28",
                "rows = 17",
                "board_height = 7",
                "board_width = 8.5",
                'dictionary = "DICT_4X4_50"',
                'units = "inch"',
                "aruco_scale = 0.75",
                "inverted = false",
                "legacy_pattern = false",
            ]
        )
    )
    charuco = Charuco.from_toml(path)
    assert charuco.dictionary == "DICT_4X4_250"
    # And the loaded board renders at the preview scale without crashing.
    assert charuco.board_img(pixmap_scale=200).size > 0


def test_project_open_preview_renders_for_hand_edited_dense_board(tmp_path: Path):
    """The #978 startup repro: a mismatched intrinsic_charuco.toml must render.

    Exercises the real preview path (repository load -> render_charuco_pixmap)
    that runs at project open. QPixmap needs a QApplication.
    """
    from PySide6.QtWidgets import QApplication

    from caliscope.gui.utils.charuco_preview import render_charuco_pixmap

    targets_dir = tmp_path / "targets"
    targets_dir.mkdir()
    (targets_dir / "intrinsic_charuco.toml").write_text(
        "\n".join(
            [
                "columns = 28",
                "rows = 17",
                "board_height = 7",
                "board_width = 8.5",
                'dictionary = "DICT_4X4_50"',
                'units = "inch"',
                "aruco_scale = 0.75",
                "inverted = false",
                "legacy_pattern = false",
            ]
        )
    )

    app = QApplication.instance() or QApplication([])
    assert app is not None

    repo = CalibrationTargetsRepository(targets_dir)
    charuco = repo.load_intrinsic_charuco()
    pixmap = render_charuco_pixmap(charuco, 200)
    assert not pixmap.isNull()


if __name__ == "__main__":
    debug_dir = Path(__file__).parent / "tmp"
    debug_dir.mkdir(parents=True, exist_ok=True)

    board = _reporter_board()
    img = board.board_img(pixmap_scale=2000)
    import cv2

    cv2.imwrite(str(debug_dir / "reporter_board.png"), img)
    print(f"marker_count={board.marker_count}, dictionary={board.dictionary}")
