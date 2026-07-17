"""Central tab titles for the main window's tab bar.

Tabs are looked up by title string (MainWindow.find_tab_index_by_title), and
a missed lookup returns -1, which Qt treats as a silent no-op in
setTabEnabled. Every load-bearing tab title must come from this enum so a
typo fails loudly at the attribute lookup instead of greying out a tab
forever. Lives in its own module because both main_widget and the views
need it (main_widget importing from a view would cycle).
"""

from enum import StrEnum


class TabName(StrEnum):
    PROJECT = "Project"
    CAMERAS = "Cameras"
    MULTI_CAMERA = "Multi-Camera"
    CALIBRATE = "Calibrate"
    RECONSTRUCTION = "Reconstruction"
