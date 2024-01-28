import caliscope.logger


from pathlib import Path
import pandas as pd
from caliscope.trackers.tracker_enum import TrackerEnum

from PySide6.QtCore import Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QMessageBox,
    QWidget,
    QComboBox,
    QListWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from caliscope.post_processing.blender_tools import generate_metarig_config
from caliscope.controller import Controller
from caliscope.configurator import Configurator
from caliscope.gui.vizualize.playback_triangulation_widget import (
    PlaybackTriangulationWidget,
)
logger = caliscope.logger.get(__name__)

class PostProcessingWidget(QWidget):
    processing_complete = Signal()

    def __init__(self, controller:Controller):
        super(PostProcessingWidget, self).__init__()
        self.controller = controller 
        self.config = self.controller.config

        self.sync_index_cursors = {}  # track where the slider is for each playback...
        self.recording_folders = QListWidget()
        self.update_recording_folders()

        self.vis_widget = PlaybackTriangulationWidget(self.controller.camera_array)  

        self.tracker_combo = QComboBox()
        self.vizualizer_title = QLabel()

        # Add items to the combo box using the name attribute of the TrackerEnum
        for tracker in TrackerEnum:
            if tracker.name != "CHARUCO":
                self.tracker_combo.addItem(tracker.name, tracker)


        self.open_folder_btn = QPushButton("&Open Folder")
        self.process_current_btn = QPushButton("&Process")
        self.generate_metarig_config_btn = QPushButton("Generate Metarig Config")
        self.refresh_visualizer() # must happen before placement to create vis_widget and vizualizer_title
        self.place_widgets()
        self.connect_widgets()
        

    def set_current_xyz(self):

        if self.xyz_processed_path.exists():
            self.vis_widget.update_motion_trial(self.xyz_processed_path)
            # # confirm that there are some triangulated values to observe
            # xyz = pd.read_csv(self.xyz_processed_path)
            # if xyz.shape[0] != 0:
            #     logger.info(f"Setting xyz display coordinates to those stored in {self.xyz_processed_path}")
            #     self.xyz = xyz
            # else:
            #     logger.info("Not enough data to triangulate points")
            #     QMessageBox.warning(self, "Warning", f"The {self.active_tracker_enum.name} tracker did not identify sufficient points for triangulation to occur for recordings stored in:\n{self.active_recording_path}.") # show a warning dialog
            #     self.xyz = None
        else:
            logger.info(f"No points displayed; Nothing stored in {self.xyz_processed_path}")
            # self.xyz = None

            # check if there aren't any points to track and warn about that
            if self.xy_base_path.exists():
                xy = pd.read_csv(self.xy_base_path)
                if xy.shape[0] == 0:
                    logger.info("No points tracked")
                    QMessageBox.warning(self, "Warning", f"The {self.active_tracker_enum.name} tracker did not identify any points to track in recordings stored in:\n{self.active_recording_path}.") # show a warning dialog


    def update_recording_folders(self):
        # this check here is an artifact of the way that the main widget handles refresh
        self.recording_folders.clear()
        # create list of recording directories
        dir_list = self.controller.workspace_guide.valid_recording_dirs()

        # add each folder to the QListWidget
        for folder in dir_list:
            self.recording_folders.addItem(folder)
        
        if len(dir_list)>0:
            self.recording_folders.setCurrentRow(0)

    @property
    def processed_subfolder(self):
        subfolder = Path(
            self.controller.workspace_guide.recording_dir,
            self.recording_folders.currentItem().text(),
            self.tracker_combo.currentData().name,
        )
        return subfolder

    @property
    def xyz_processed_path(self):
        file_name = f"xyz_{self.tracker_combo.currentData().name}.csv"
        result = Path(self.processed_subfolder, file_name)
        return result

    @property
    def archived_config_path(self):
        return Path(self.processed_subfolder, "config.toml")
        

    @property
    def xy_base_path(self):
        file_name = f"xy_{self.tracker_combo.currentData().name}.csv"
        result = Path(self.processed_subfolder, file_name)
        return result

    @property
    def active_tracker_enum(self):
        return self.tracker_combo.currentData()
        
    @property
    def metarig_config_path(self):
        file_name = f"metarig_config_{self.tracker_combo.currentData().name}.json"
        result = Path(self.processed_subfolder, file_name)
        return result
        

    @property
    def active_folder(self):
        if self.recording_folders.count() == 0:
            active_folder = None
        elif self.recording_folders.currentItem() is None:
            self.recording_folders.setCurrentRow(0)
            active_folder: str = self.recording_folders.currentItem().text()
        else:
            active_folder: str = self.recording_folders.currentItem().text()
            
        return active_folder

    @property
    def active_recording_path(self)-> Path:
        p = Path(self.controller.workspace_guide.recording_dir, self.active_folder)
        logger.info(f"Active recording path is {p}")
        return p        
        
    @property
    def viz_title_html(self):
        if self.xyz_processed_path.exists():
            suffix = "(x,y,z) estimates"
        else:
            suffix = "(no processed data)"

        title = f"<div align='center'><b>{self.tracker_combo.currentData().name.title()} Tracker: {self.active_folder} {suffix} </b></div>"

        return title

    def place_widgets(self):
        self.setLayout(QHBoxLayout())
        self.left_vbox = QVBoxLayout()
        self.right_vbox = QVBoxLayout()
        self.button_hbox = QHBoxLayout()

        self.layout().addLayout(self.left_vbox)

        self.left_vbox.addWidget(self.recording_folders)
        self.left_vbox.addWidget(self.open_folder_btn)
        self.left_vbox.addWidget(self.tracker_combo)
        self.button_hbox.addWidget(self.process_current_btn)
        self.button_hbox.addWidget(self.generate_metarig_config_btn)
        self.left_vbox.addLayout(self.button_hbox)

        self.layout().addLayout(self.right_vbox, stretch=2)
        self.right_vbox.addWidget(self.vizualizer_title)
        self.right_vbox.addWidget(self.vis_widget, stretch=2)

    def connect_widgets(self):
        self.recording_folders.currentItemChanged.connect(self.refresh_visualizer)
        self.tracker_combo.currentIndexChanged.connect(self.refresh_visualizer)
        self.vis_widget.slider.valueChanged.connect(self.store_sync_index_cursor)
        self.process_current_btn.clicked.connect(self.process_current)
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.generate_metarig_config_btn.clicked.connect(self.create_metarig_config)

        self.controller.post_processing_complete.connect(self.enable_all_inputs)
        self.controller.post_processing_complete.connect(self.refresh_visualizer)

    def store_sync_index_cursor(self, cursor_value):
        if self.xyz_processed_path.exists():
            self.sync_index_cursors[self.xyz_processed_path] = cursor_value

    def open_folder(self):
        """Opens the currently active folder in a system file browser"""
        if self.active_folder is not None:
            folder_path = Path(self.controller.workspace_guide.recording_dir, self.active_folder)
            url = QUrl.fromLocalFile(str(folder_path))
            QDesktopServices.openUrl(url)
        else:
            logger.warn("No folder selected")

    def process_current(self):
        """
        
        This needs to get pushed into the controller layer
        """
        recording_path = Path(self.controller.workspace_guide.recording_dir, self.active_folder)
        logger.info(f"Beginning processing of recordings at {recording_path}")
        tracker_enum = self.tracker_combo.currentData()
        logger.info(f"(x,y) tracking will be applied using {tracker_enum.name}")
        recording_config_toml  = Path(recording_path,"config.toml")
        logger.info(f"Camera data based on config file saved to {recording_config_toml}")

        self.controller.process_recordings(recording_path, tracker_enum)

    def active_tracker_config_path(self):
        """
        This will exist of the tracker has been processed. It should have a camera array within it 
        that can be displayed to the user and will align 
        """
        return Path(self.active_recording_path,self.active_tracker_enum.name, "config.toml")
        
    def refresh_visualizer(self):
        logger.info("Refreshing vizualizer within post_processing widget")
        
        if self.archived_config_path.exists():  # processing has been done and their is a camera array that can be loaded
            stored_config = Configurator(self.archived_config_path.parent)
            presented_camera_array = stored_config.get_camera_array()
        else:
            presented_camera_array = self.controller.camera_array

        self.vis_widget.update_camera_array(presented_camera_array)

        
        self.set_current_xyz()
        self.vizualizer_title.setText(self.viz_title_html)
        self.update_enabled_disabled()
        self.update_slider_position()

    def disable_all_inputs(self):
        """used to toggle off all inputs will processing is going on"""
        self.recording_folders.setEnabled(False)
        self.tracker_combo.setEnabled(False)
        # self.export_btn.setEnabled(False)
        self.process_current_btn.setEnabled(False)
        self.vis_widget.slider.setEnabled(False)

    def enable_all_inputs(self):
        """
        after processing completes, swithes everything on again,
        but fine tuning of enable/disable will happen with self.update_enabled_disabled
        """
        self.recording_folders.setEnabled(True)
        self.tracker_combo.setEnabled(True)
        self.process_current_btn.setEnabled(True)
        self.vis_widget.slider.setEnabled(True)

    def update_enabled_disabled(self):
        # set availability of metarig generation 
        logger.info("Checking if metarig config can be created...")
        tracker = self.tracker_combo.currentData().value()
        logger.info(tracker)
        if (tracker.metarig_mapped and self.xyz_processed_path.exists() and not self.metarig_config_path.exists()):
            self.generate_metarig_config_btn.setEnabled(True)
            self.generate_metarig_config_btn.setToolTip("Creation of metarig configuration file is now available")
        else:
            self.generate_metarig_config_btn.setEnabled(False)
        
        if not tracker.metarig_mapped:
            self.generate_metarig_config_btn.setToolTip("Tracker is not set up to scale to a metarig")
        elif self.metarig_config_path.exists():
            self.generate_metarig_config_btn.setToolTip("The Metarig configuration json file has already been created.Check the tracker subfolder in the recording directory.")
        elif not self.xyz_processed_path.exists():
            self.generate_metarig_config_btn.setToolTip("Must process recording to create xyz estimates for metarig configuration")
        else:
            self.generate_metarig_config_btn.setToolTip("Click to create a file in the tracker subfolder that can be used to scale a Blender metarig")
        # set availability of Proecssing and slider                
        if self.xyz_processed_path.exists():
            self.process_current_btn.setEnabled(False)
            self.vis_widget.slider.setEnabled(True)
        elif self.xy_base_path.exists() and not self.xyz_processed_path.exists():
            # nothing available to triangulate
            self.process_current_btn.setEnabled(False)
            self.vis_widget.slider.setEnabled(False)
            
        else:
            self.process_current_btn.setEnabled(True)
            self.vis_widget.slider.setEnabled(False)

    def update_slider_position(self):
        # update slider value to stored value if it exists
        if self.xyz_processed_path in self.sync_index_cursors.keys():
            active_sync_index = self.sync_index_cursors[self.xyz_processed_path]
            self.vis_widget.slider.setValue(active_sync_index)
            self.vis_widget.visualizer.display_points(active_sync_index)
        else:
            pass


    def create_metarig_config(self):
        logger.info(f"Beginning metarig_config creation in {self.processed_subfolder}")
        tracker_enum = self.tracker_combo.currentData()
        xyz_csv_path = Path(self.processed_subfolder, f"xyz_{tracker_enum.name}_labelled.csv")
        generate_metarig_config(tracker_enum, xyz_csv_path)
        self.update_enabled_disabled()

