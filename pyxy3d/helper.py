import caliscope.logger

from pathlib import Path
import shutil
logger = caliscope.logger.get(__name__)

def copy_contents(src_folder, dst_folder):
    """
    Helper function to port a test case data folder over to a temp directory 
    used for testing purposes so that the test case data doesn't get overwritten
    """
    src_path = Path(src_folder)
    dst_path = Path(dst_folder)


    if dst_path.exists():
        shutil.rmtree(dst_path)
        
    # Create the destination folder if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=False)


        
    for item in src_path.iterdir():
        # Construct the source and destination paths
        src_item = src_path / item
        dst_item = dst_path / item.name


        # Copy file or directory
        if src_item.is_file():
            logger.info(f"Copying file at {src_item} to {dst_item}")
            shutil.copy2(src_item, dst_item)  # Copy file preserving metadata

        elif src_item.is_dir():
            logger.info(f"Copying directory at {src_item} to {dst_item}")
            shutil.copytree(src_item, dst_item)
