"""Unit tests for ONNX model download and extraction.

All tests use file:// URLs pointing to local temporary files — no network access.
"""

import hashlib
import zipfile
from pathlib import Path

import pytest

from caliscope.trackers.model_card import ModelCard
from caliscope.trackers.model_download import download_and_extract_model


def _make_card(
    source_url: str | None = None,
    extraction: str | None = None,
    model_path: Path = Path("test_model.onnx"),
    sha256: str | None = None,
) -> ModelCard:
    """Create a minimal ModelCard for testing download functionality."""
    return ModelCard(
        name="Test Model",
        model_path=model_path,
        format="simcc",
        input_width=48,
        input_height=64,
        confidence_threshold=0.3,
        point_name_to_id={"nose": 0},
        wireframe=None,
        source_url=source_url,
        extraction=extraction,
        sha256=sha256,
    )


def _create_test_zip(zip_path: Path, onnx_content: bytes, nested_name: str = "model_dir/end2end.onnx"):
    """Create a test zip with an onnx file at the given path inside the archive."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(nested_name, onnx_content)


@pytest.fixture
def fixture_onnx() -> bytes:
    """Load the 3KB fixture ONNX model as bytes."""
    fixture_path = Path(__file__).parent / "fixtures" / "onnx" / "simcc_3pt.onnx"
    return fixture_path.read_bytes()


def test_download_zip_end2end(tmp_path: Path, fixture_onnx: bytes):
    """Happy path: download and extract from a zip containing end2end.onnx."""
    # Create a temporary zip file containing the ONNX model
    zip_path = tmp_path / "model_archive.zip"
    _create_test_zip(zip_path, fixture_onnx, "some_dir/end2end.onnx")

    # Create a card pointing to the zip via file:// URL
    card = _make_card(
        source_url=zip_path.as_uri(),
        extraction="zip_end2end",
        model_path=Path("my_model.onnx"),
    )

    # Download and extract
    target_dir = tmp_path / "models"
    result_path = download_and_extract_model(card, target_dir)

    # Verify
    assert result_path == target_dir / "my_model.onnx"
    assert result_path.exists()
    assert result_path.stat().st_size > 0
    assert result_path.read_bytes() == fixture_onnx


def test_download_direct(tmp_path: Path, fixture_onnx: bytes):
    """Happy path: download a direct .onnx file."""
    # Create a temporary ONNX file
    onnx_path = tmp_path / "source_model.onnx"
    onnx_path.write_bytes(fixture_onnx)

    # Create a card pointing to the ONNX file via file:// URL
    card = _make_card(
        source_url=onnx_path.as_uri(),
        extraction="direct",
        model_path=Path("my_model.onnx"),
    )

    # Download
    target_dir = tmp_path / "models"
    result_path = download_and_extract_model(card, target_dir)

    # Verify
    assert result_path == target_dir / "my_model.onnx"
    assert result_path.exists()
    assert result_path.read_bytes() == fixture_onnx


def test_sha256_verification_pass(tmp_path: Path, fixture_onnx: bytes):
    """SHA-256 verification succeeds when hash matches."""
    # Create a zip file and compute its hash
    zip_path = tmp_path / "model_archive.zip"
    _create_test_zip(zip_path, fixture_onnx)
    correct_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    # Create a card with the correct hash
    card = _make_card(
        source_url=zip_path.as_uri(),
        extraction="zip_end2end",
        model_path=Path("my_model.onnx"),
        sha256=correct_hash,
    )

    # Download and extract — should succeed
    target_dir = tmp_path / "models"
    result_path = download_and_extract_model(card, target_dir)

    # Verify the file was created (no exception raised)
    assert result_path.exists()


def test_sha256_verification_fail(tmp_path: Path, fixture_onnx: bytes):
    """SHA-256 verification fails when hash doesn't match."""
    # Create a zip file
    zip_path = tmp_path / "model_archive.zip"
    _create_test_zip(zip_path, fixture_onnx)

    # Create a card with an incorrect hash
    wrong_hash = "0" * 64
    card = _make_card(
        source_url=zip_path.as_uri(),
        extraction="zip_end2end",
        model_path=Path("my_model.onnx"),
        sha256=wrong_hash,
    )

    # Download should raise ValueError
    target_dir = tmp_path / "models"
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        download_and_extract_model(card, target_dir)


def test_missing_end2end_in_zip(tmp_path: Path, fixture_onnx: bytes):
    """Extraction fails when zip doesn't contain end2end.onnx."""
    # Create a zip with a different file name
    zip_path = tmp_path / "model_archive.zip"
    _create_test_zip(zip_path, fixture_onnx, "some_dir/wrong_name.onnx")

    card = _make_card(
        source_url=zip_path.as_uri(),
        extraction="zip_end2end",
        model_path=Path("my_model.onnx"),
    )

    # Download should raise FileNotFoundError
    target_dir = tmp_path / "models"
    with pytest.raises(FileNotFoundError, match="end2end.onnx not found"):
        download_and_extract_model(card, target_dir)


def test_no_source_url(tmp_path: Path):
    """Raises ValueError when ModelCard has no source_url."""
    card = _make_card(source_url=None, extraction="direct")

    target_dir = tmp_path / "models"
    with pytest.raises(ValueError, match="no source_url"):
        download_and_extract_model(card, target_dir)


def test_cancellation(tmp_path: Path, fixture_onnx: bytes):
    """Download is cancelled when cancellation_check returns True."""
    # Create a source file
    onnx_path = tmp_path / "source_model.onnx"
    onnx_path.write_bytes(fixture_onnx)

    card = _make_card(
        source_url=onnx_path.as_uri(),
        extraction="direct",
        model_path=Path("my_model.onnx"),
    )

    # Cancellation check that immediately returns True
    def always_cancelled() -> bool:
        return True

    target_dir = tmp_path / "models"
    with pytest.raises(InterruptedError):
        download_and_extract_model(card, target_dir, cancellation_check=always_cancelled)


def test_progress_callback_called(tmp_path: Path, fixture_onnx: bytes):
    """Progress callback is called during download with bytes_downloaded > 0."""
    # Create a source file
    onnx_path = tmp_path / "source_model.onnx"
    onnx_path.write_bytes(fixture_onnx)

    card = _make_card(
        source_url=onnx_path.as_uri(),
        extraction="direct",
        model_path=Path("my_model.onnx"),
    )

    # Track progress calls
    progress_calls: list[tuple[int, int]] = []

    def track_progress(bytes_downloaded: int, total_bytes: int) -> None:
        progress_calls.append((bytes_downloaded, total_bytes))

    target_dir = tmp_path / "models"
    download_and_extract_model(card, target_dir, progress_callback=track_progress)

    # Verify callback was called at least once with bytes_downloaded > 0
    assert len(progress_calls) > 0
    assert any(bytes_down > 0 for bytes_down, _ in progress_calls)


if __name__ == "__main__":
    """Debug harness for manual testing and debugging."""
    import tempfile

    debug_dir = Path(__file__).parent / "tmp" / "model_download"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Debug output directory: {debug_dir}")

    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "onnx" / "simcc_3pt.onnx"
    fixture_bytes = fixture_path.read_bytes()
    print(f"Loaded fixture: {len(fixture_bytes)} bytes")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Test 1: zip_end2end extraction
        print("\n=== Test 1: zip_end2end ===")
        zip_path = tmp_path / "test_model.zip"
        _create_test_zip(zip_path, fixture_bytes)
        print(f"Created zip: {zip_path}")

        card = _make_card(
            source_url=zip_path.as_uri(),
            extraction="zip_end2end",
            model_path=Path("extracted_model.onnx"),
        )

        result = download_and_extract_model(card, debug_dir / "test1")
        print(f"Extracted to: {result}")
        print(f"File exists: {result.exists()}, size: {result.stat().st_size}")

        # Test 2: direct download
        print("\n=== Test 2: direct ===")
        direct_path = tmp_path / "direct_model.onnx"
        direct_path.write_bytes(fixture_bytes)

        card = _make_card(
            source_url=direct_path.as_uri(),
            extraction="direct",
            model_path=Path("direct_model.onnx"),
        )

        result = download_and_extract_model(card, debug_dir / "test2")
        print(f"Downloaded to: {result}")
        print(f"File exists: {result.exists()}, size: {result.stat().st_size}")

        # Test 3: SHA-256 verification
        print("\n=== Test 3: SHA-256 verification ===")
        zip_path_hash = tmp_path / "test_model_hash.zip"
        _create_test_zip(zip_path_hash, fixture_bytes)
        correct_hash = hashlib.sha256(zip_path_hash.read_bytes()).hexdigest()
        print(f"Computed SHA-256: {correct_hash}")

        card = _make_card(
            source_url=zip_path_hash.as_uri(),
            extraction="zip_end2end",
            model_path=Path("verified_model.onnx"),
            sha256=correct_hash,
        )

        result = download_and_extract_model(card, debug_dir / "test3")
        print(f"Verified and extracted to: {result}")

        # Test 4: Progress tracking
        print("\n=== Test 4: Progress tracking ===")
        progress_log: list[tuple[int, int]] = []

        def log_progress(bytes_down: int, total: int) -> None:
            progress_log.append((bytes_down, total))

        result = download_and_extract_model(card, debug_dir / "test4", progress_callback=log_progress)
        print(f"Progress calls: {len(progress_log)}")
        for i, (down, total) in enumerate(progress_log):
            print(f"  Call {i}: {down} / {total} bytes")

    print("\n=== All debug tests completed ===")
