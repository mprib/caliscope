import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def copy_contents_to_clean_dest(src_folder: str | Path, dst_folder: str | Path) -> None:
    """
    **DESTRUCTIVE OPERATION**: Copies src_folder to dst_folder,
    FIRST DELETING dst_folder if it exists.

    Use this in tests with pytest's tmp_path fixture to ensure
    parallel-safe test isolation.

    Args:
        src_folder: Source directory to copy from
        dst_folder: Destination directory (will be deleted and recreated!)
    """
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)

    if dst_path.exists():
        logger.debug(f"Removing existing destination: {dst_path}")
        shutil.rmtree(dst_path)

    # Create the destination folder
    dst_path.mkdir(parents=True, exist_ok=False)

    for item in src_path.iterdir():
        src_item = src_path / item
        dst_item = dst_path / item.name

        if src_item.is_file():
            logger.debug(f"Copying file: {src_item} -> {dst_item}")
            shutil.copy2(src_item, dst_item)
        elif src_item.is_dir():
            logger.debug(f"Copying directory: {src_item} -> {dst_item}")
            shutil.copytree(src_item, dst_item)
