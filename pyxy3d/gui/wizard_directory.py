import pyxy3d.logger

logger = pyxy3d.logger.get(__name__)

import os
from pathlib import Path


import sys
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QFileDialog,
    QWidget,
    QDialog,
    QMessageBox,
    QLabel,
    QVBoxLayout,
    QRadioButton,
    QButtonGroup,
)
from pyxy3d import __app_dir__, __root__

from pyxy3d.session import Session

class WizardDirectory(QWidget):
    
    isComplete = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()

        text = QLabel(
            "This wizard will help you calibrate your camera system. \nWill you be starting with a previous configuration \nor creating a new one?"
        )

        # two options: new config from scratch, or begin with previous (camera config/charuco the same)
        self.from_previous_radio = QRadioButton("From Previous Configuration")
        self.from_previous_radio.clicked.connect(self.click_from_previous)
        self.create_new_radio = QRadioButton("Create New Configuration")
        self.create_new_radio.clicked.connect(self.click_new)


        # button groups only allow one at a time selected
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.from_previous_radio)
        self.button_group.addButton(self.create_new_radio)

        # create the general layout
        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)
        self.vbox.addWidget(text)

        self.vbox.addWidget(self.from_previous_radio)

        self.original_path = DirectorySelector(self, "Original Folder", self.check_old_directory_validity)
        self.vbox.addWidget(self.original_path)
        self.original_path.setHidden(True)
        self.modified_path = DirectorySelector(self, "New Folder")
        self.vbox.addWidget(self.modified_path)
        self.modified_path.setHidden(True)
        self.vbox.addWidget(self.create_new_radio)
        
        self.new_path = DirectorySelector(self, "Folder")
        self.vbox.addWidget(self.new_path)
        self.new_path.setHidden(True)
        
        # create the button that will initiate the wizard once file(s) chosen 
        self.launch_wizard_btn = QPushButton("Launch Calibration Wizard")
            
        self.launch_wizard_btn.clicked.connect(self.launch_wizard)
        self.launch_wizard_btn.setEnabled(False)
        
        self.vbox.addWidget(self.launch_wizard_btn)
        
    def launch_wizard(self):
        # where you'll need to link up the next dialog in the chain
        
        print(self.original_path.textbox.text())
        print(self.modified_path.textbox.text())
        print(self.new_path.textbox.text())
        print("Time to move on...")
        
    def click_from_previous(self):
        print(self.button_group.checkedButton().text())

        self.original_path.setHidden(False)
        self.modified_path.setHidden(False)
        self.new_path.setHidden(True)
        self.isComplete.emit(self.check_complete())
       
        
    def click_new(self):
        print(self.button_group.checkedButton().text())
        self.original_path.setHidden(True)
        self.modified_path.setHidden(True)
        self.new_path.setHidden(False)
        self.isComplete.emit(self.check_complete())
        
    def check_complete(self) -> bool:
        print("Checking if complete")
        if self.create_new_radio.isChecked():
            print("new calibration checked")
            if os.path.exists(self.new_path.textbox.text()):
                self.session_path = self.new_path.textbox.text()
                print("Is complete")
                self.launch_wizard_btn.setEnabled(True)
                return True

            self.launch_wizard_btn.setEnabled(False)
            return False
            
        if self.from_previous_radio.isChecked():
            print("from previous checked")
            if os.path.exists(Path(self.original_path.textbox.text(), "config.toml")) and os.path.exists(self.modified_path.textbox.text()):
                self.session_path = self.original_path.textbox.text()
                print("Is complete")
                self.launch_wizard_btn.setEnabled(True)
                return True
            self.launch_wizard_btn.setEnabled(False)
            return False
        
        self.launch_wizard_btn.setEnabled(False)
        return False

    def check_old_directory_validity(self, fname):
         
        old_directory_good = os.path.isfile(Path(fname,"config.toml"))
        
        if old_directory_good:
            message = "NA"
        else:
            message = "Folder does not contain `config.toml`" 
        
        return old_directory_good, message
                             
class DirectorySelector(QWidget):
    def __init__(self, qwizard_page, button_text, validity_check = None):
        super().__init__()
        self.textbox = QLineEdit()
        self.textbox.setEnabled(False)
        self.button = QPushButton(button_text)
        self.button.clicked.connect(self.select_directory)
        self.qwizard_page = qwizard_page
        self.validity_check = validity_check # validity check must contain a tuple of (bool, str) which is (validity, message)
        
        layout = QHBoxLayout()
        layout.addWidget(self.textbox)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def select_directory(self):

        fname = QFileDialog.getExistingDirectory(
            # self, "Select Folder", str(__app_dir__)
            self, "Select Folder", str(Path(__root__, "tests"))   # done for testing to track impact on config Easier
        )
        
        if self.validity_check is None:
            self.textbox.setText(fname)
            self.parent().isComplete.emit(self.parent().check_complete())
        else:
            if self.validity_check(fname)[0]:
                self.textbox.setText(fname)
                self.parent().isComplete.emit(self.parent().check_complete())
            else:
                logger.info(f"Invalid: {self.validity_check(fname)[1]}") 
                message_box = QMessageBox()
                message_box.setWindowTitle("Invalid Directory")
                message_box.setText(self.validity_check(fname)[1])
                # message_box.setStandardButtons(QMessageBox.Ok)
                message_box.exec()

            
            
if __name__ == "__main__":

    app = QApplication(sys.argv)
    # wizard = FolderSelectWizard()

    # wizard.show()
    wizard_intro = WizardDirectory()
    wizard_intro.show()
    sys.exit(app.exec())


