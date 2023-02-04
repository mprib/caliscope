import calicam.logger

logger = calicam.logger.get(__name__)

import os

import sys
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget
from PyQt6.QtWidgets import (
    QApplication,
    QWizard,
    QWizardPage,
    QDialog,
    QLabel,
    QVBoxLayout,
    QRadioButton,
    QButtonGroup,
)
from calicam import __app_dir__

from calicam.session import Session

class FolderSelectWizard(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Wizard")

        self.intro_page = WizardIntro()
        self.addPage(self.intro_page)
    
        self.launch_wizard_btn = QPushButton("Launch Calibration Session")
        self.launch_wizard_btn.setEnabled(False)
        self.setButton(self.WizardButton.CustomButton1, self.launch_wizard_btn)
        self.setButtonLayout([self.WizardButton.CustomButton1])

class WizardIntro(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Something old or something new?")

        text = QLabel(
            "This wizard will help you calibrate your camera system. \nWill you be starting with a previous configuration \nor creating a new one?"
        )

        self.from_previous_radio = QRadioButton("From Previous Configuration")
        self.from_previous_radio.clicked.connect(self.click_from_previous)
        self.create_new_radio = QRadioButton("Create New Configuration")
        self.create_new_radio.clicked.connect(self.click_new)

        # button groups only allow one at a time selected
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.from_previous_radio)
        self.button_group.addButton(self.create_new_radio)

        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)
        self.vbox.addWidget(text)

        self.vbox.addWidget(self.from_previous_radio)

        self.original_path = DirectorySelector(self, "Original Config")
        self.vbox.addWidget(self.original_path)
        self.original_path.setHidden(True)
        self.modified_path = DirectorySelector(self, "New Config")
        self.vbox.addWidget(self.modified_path)
        self.modified_path.setHidden(True)
        self.vbox.addWidget(self.create_new_radio)
        
        self.new_path = DirectorySelector(self, "New Config")
        self.vbox.addWidget(self.new_path)
        self.new_path.setHidden(True)

    def click_from_previous(self):
        print(self.button_group.checkedButton().text())

        self.original_path.setHidden(False)
        self.modified_path.setHidden(False)
        self.new_path.setHidden(True)
        self.completeChanged.emit()
        
    def click_new(self):
        print(self.button_group.checkedButton().text())
        self.original_path.setHidden(True)
        self.modified_path.setHidden(True)
        self.new_path.setHidden(False)
        self.completeChanged.emit()
        
    def isComplete(self) -> bool:
        print("Checking if complete")
        if self.create_new_radio.isChecked():
            if os.path.exists(self.new_path.textbox.text()):
                self.wizard().session_path = self.new_path.textbox.text()
                self.wizard().launch_wizard_btn.setEnabled(True)
                print("Is complete")
                return True
            

        if self.from_previous_radio.isChecked():
            if os.path.exists(self.original_path.textbox.text()):
                if os.path.exists(self.modified_path.textbox.text()):
                    self.wizard().session_path = self.original_path.textbox.text()
                    self.wizard().launch_wizard_btn.setEnabled(True)
                    print("Is complete")
                    return True

        self.wizard().launch_wizard_btn.setEnabled(False)
        return False

class DirectorySelector(QWidget):
    def __init__(self, qwizard_page, button_text):
        super().__init__()
        self.textbox = QLineEdit()
        self.textbox.setEnabled(False)
        self.button = QPushButton(button_text)
        self.button.clicked.connect(self.select_directory)
        self.qwizard_page = qwizard_page
        
        layout = QHBoxLayout()
        layout.addWidget(self.textbox)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def select_directory(self):

        fname = QFileDialog.getExistingDirectory(
            self, "Select Folder", str(__app_dir__)
        )
        self.textbox.setText(fname)
        print(fname)
        self.qwizard_page.completeChanged.emit()

if __name__ == "__main__":

    app = QApplication(sys.argv)
    wizard = FolderSelectWizard()
    wizard.show()
    sys.exit(app.exec())


