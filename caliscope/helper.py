import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pywinctl as pwc

import caliscope.logger

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


def view_df(df: pd.DataFrame, name: str = "df_view"):
    """
    Saves a DataFrame to a temporary Excel file and opens it.

    If a window for a previous file with the same name is open, it will
    be closed before the new one is opened.

    Args:
        df: The pandas DataFrame to view.
        name: The base name for the temporary file (without extension).

    Requires `openpyxl` and `pygetwindow`:
    `pip install openpyxl pygetwindow`
    """

    filename = f"{name}.xlsx"
    filepath = os.path.join(tempfile.gettempdir(), filename)

    # --- New Logic to Close Existing Windows ---
    if os.path.exists(filepath):
        try:
            # Find windows whose titles contain the filename. This is more robust
            # than an exact match (e.g., handles "df_view.xlsx - Excel").
            open_windows = pwc.getWindowsWithTitle(filename)
            for win in open_windows:
                print(f"Closing existing window: '{win.title}'")
                win.close()

            # Wait a moment for the OS to close the window and release the file lock.
            if open_windows:
                time.sleep(0.5)

        except Exception as e:
            # Catch exceptions if the window is unresponsive or already closed.
            print(f"Could not close existing window (it may be closed already): {e}")

    # --- Original Logic to Save and Open ---
    print(f"Opening DataFrame in Excel... saved to: {filepath}")

    try:
        # Save the DataFrame to the Excel file.
        df.to_excel(filepath, index=False)
    except PermissionError:
        print("\nPermissionError: Could not write to file. It might still be open or locked.")
        print("Please close the file manually and try again.")
        return

    # Open the file with the default system application.
    try:
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", filepath], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", filepath], check=True)
    except Exception as e:
        print(f"Error: Could not open the file automatically: {e}")
        print(f"Please open it manually: {filepath}")
