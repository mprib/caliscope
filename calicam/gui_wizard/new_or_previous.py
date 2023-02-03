import sys
from PyQt6.QtWidgets import QApplication, QWizard, QWizardPage, QLabel, QVBoxLayout, QRadioButton, QButtonGroup, QLineEdit, QFileDialog, QHBoxLayout, QPushButton

app = QApplication(sys.argv)

wizard = QWizard()
wizard.setWindowTitle("Simple Wizard")

page1 = QWizardPage()
page1.setTitle("Introduction")

label1 = QLabel("This wizard will help you register for the wizard tournament.")

from_scratch_radio = QRadioButton("From Scratch")
from_previous_radio = QRadioButton("From Previous")

button_group = QButtonGroup()
button_group.addButton(from_scratch_radio)
button_group.addButton(from_previous_radio)

page1.setLayout(QHBoxLayout())

destination_line = QLineEdit()
destination_line.setEnabled(False)
destination_line.setPlaceholderText("Destination")

def choose_destination():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Directory Files (*)", options=options)
    if file_name:
        destination_line.setText(file_name)

destination_button = QPushButton("Choose Destination")
destination_button.clicked.connect(choose_destination)
destination_button.setEnabled(False)

destination_layout = QHBoxLayout()
destination_layout.addWidget(destination_line)
destination_layout.addWidget(destination_button)

origin_line = QLineEdit()
origin_line.setEnabled(False)
origin_line.setPlaceholderText("Origin")

destination2_line = QLineEdit()
destination2_line.setEnabled(False)
destination2_line.setPlaceholderText("Destination")

def choose_origin():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Directory Files (*)", options=options)
    if file_name:
        origin_line.setText(file_name)

def choose_destination2():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    file_name, _ = QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","All Files (*);;Directory Files (*)", options=options)
    if file_name:
        destination2_line.setText(file_name)

origin_button = QPushButton("Choose Origin")
origin_button.clicked.connect(choose_origin)
origin_button.setEnabled(False)

destination2_button = QPushButton("Choose Destination")
# destination2_button.clicked.connect(choose_
wizard.show()
sys.exit(app.exec())
