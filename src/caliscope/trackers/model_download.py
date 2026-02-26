"""ONNX model download and extraction.

Pure domain module — NO Qt dependency. Uses only stdlib for downloading
and extracting ONNX models from remote sources.
"""

import hashlib
import logging
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from caliscope.trackers.model_card import ModelCard

logger = logging.getLogger(__name__)


def _download_to_temp(
    url: str,
    temp_dir: Path,
    progress_callback: Callable[[int, int], None] | None,
    cancellation_check: Callable[[], bool] | None,
) -> Path:
    """Download a file from URL to temporary directory.

    Args:
        url: Source URL to download from
        temp_dir: Temporary directory to save the file
        progress_callback: Called with (bytes_downloaded, total_bytes).
            total_bytes is -1 if server doesn't send Content-Length.
        cancellation_check: If returns True, abort and raise InterruptedError

    Returns:
        Path to the downloaded file in temp_dir

    Raises:
        ConnectionError: On network failure
        InterruptedError: If cancelled via cancellation_check
    """
    try:
        response = urllib.request.urlopen(url)  # noqa: S310 — URL comes from TOML card, not user input
    except urllib.error.HTTPError as e:
        # HTTPError is a subclass of URLError — must be caught first
        raise ConnectionError(f"HTTP error {e.code} when downloading {url}: {e}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to {url}: {e}") from e

    # Extract filename from URL
    filename = Path(url).name
    output_path = temp_dir / filename

    # Get total size if available
    content_length = response.headers.get("Content-Length")
    total_bytes = int(content_length) if content_length else -1

    logger.debug(f"Downloading {url} to {output_path} (size: {total_bytes} bytes)")

    bytes_downloaded = 0
    chunk_size = 8192

    try:
        with open(output_path, "wb") as f:
            while True:
                # Check for cancellation
                if cancellation_check and cancellation_check():
                    logger.debug("Download cancelled by user")
                    raise InterruptedError("Download cancelled")

                chunk = response.read(chunk_size)
                if not chunk:
                    break

                f.write(chunk)
                bytes_downloaded += len(chunk)

                # Report progress
                if progress_callback:
                    progress_callback(bytes_downloaded, total_bytes)

        logger.debug(f"Download complete: {output_path} ({bytes_downloaded} bytes)")
        return output_path

    finally:
        response.close()


def _verify_sha256(file_path: Path, expected_hash: str) -> None:
    """Verify SHA-256 hash of a file.

    Args:
        file_path: Path to file to verify
        expected_hash: Expected SHA-256 hash (hex string)

    Raises:
        ValueError: If hash doesn't match
    """
    hasher = hashlib.sha256()
    chunk_size = 8192

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    actual_hash = hasher.hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 verification failed for {file_path.name}.\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )

    logger.debug(f"SHA-256 verified: {file_path.name}")


def _extract_zip_end2end(zip_path: Path, target_name: str) -> Path:
    """Extract end2end.onnx from a zip file and rename it.

    Searches for any file named 'end2end.onnx' at any nesting depth within
    the zip archive.

    Args:
        zip_path: Path to the zip file
        target_name: Desired name for the extracted file

    Returns:
        Path to the extracted and renamed file (in same directory as zip_path)

    Raises:
        FileNotFoundError: If end2end.onnx not found in zip
        zipfile.BadZipFile: If zip file is corrupt
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find end2end.onnx at any depth
        end2end_entry = None
        for name in zf.namelist():
            if name.endswith("end2end.onnx"):
                end2end_entry = name
                break

        if end2end_entry is None:
            raise FileNotFoundError(f"end2end.onnx not found in {zip_path.name}")

        logger.debug(f"Found {end2end_entry} in {zip_path.name}")

        # Extract to parent directory (the temp dir)
        extracted = zf.extract(end2end_entry, path=zip_path.parent)
        extracted_path = Path(extracted)

        # Rename to target name
        renamed_path = zip_path.parent / target_name
        extracted_path.rename(renamed_path)

        logger.debug(f"Extracted and renamed to {target_name}")
        return renamed_path


def download_and_extract_model(
    card: "ModelCard",
    target_dir: Path,
    progress_callback: Callable[[int, int], None] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> Path:
    """Download and extract an ONNX model from a ModelCard source.

    Downloads to a temporary directory, optionally verifies SHA-256,
    extracts if needed, then moves to the target directory.

    Args:
        card: ModelCard with source_url and extraction fields populated
        target_dir: Directory to place the final .onnx file (e.g., MODELS_DIR)
        progress_callback: Called with (bytes_downloaded, total_bytes).
            total_bytes is -1 if server doesn't send Content-Length.
        cancellation_check: Called periodically during download.
            If returns True, abort and raise InterruptedError.

    Returns:
        Path to the downloaded .onnx file in target_dir

    Raises:
        ValueError: If source_url is None or SHA-256 verification fails
        ConnectionError: On network failure
        zipfile.BadZipFile: If downloaded zip is corrupt
        FileNotFoundError: If end2end.onnx not found inside zip
        InterruptedError: If cancelled via cancellation_check
    """
    if card.source_url is None:
        raise ValueError(f"ModelCard '{card.name}' has no source_url configured")

    logger.info(f"Downloading model '{card.name}' from {card.source_url}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Download to temporary directory
        downloaded = _download_to_temp(
            card.source_url,
            tmp_path,
            progress_callback,
            cancellation_check,
        )

        # Verify hash if provided
        if card.sha256:
            logger.debug(f"Verifying SHA-256 for {downloaded.name}")
            _verify_sha256(downloaded, card.sha256)

        # Extract based on card.extraction
        target_name = card.model_path.name

        if card.extraction == "zip_end2end":
            onnx_path = _extract_zip_end2end(downloaded, target_name)
        elif card.extraction == "direct":
            onnx_path = tmp_path / target_name
            downloaded.rename(onnx_path)
        else:
            raise ValueError(f"Unknown extraction method '{card.extraction}' for model '{card.name}'")

        # Move to final destination
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / target_name

        shutil.move(str(onnx_path), str(final_path))
        logger.info(f"Model '{card.name}' downloaded successfully to {final_path}")

        return final_path
