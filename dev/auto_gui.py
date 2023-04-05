from PyQt6.QtWidgets import QApplication
from pyxy3d.gui.main import CalibrationWizard
import sys
from time import sleep
from pyxy3d import __root__
from pathlib import Path

config_path = Path(__root__, "tests", "217")
    
    
app = QApplication(sys.argv)
window = CalibrationWizard()

# open in a session already so you don't have to go through the menu each time
# window.open_session(config_path)
window.wizard_directory.from_previous_radio.click()
window.wizard_directory.from_previous_radio.setChecked(True)
window.wizard_directory.launch_wizard_btn.setEnabled(True)
window.wizard_directory.original_path.textbox.setText(str(config_path))
window.wizard_directory.modified_path.textbox.setText(str(config_path))
window.wizard_directory.launch_wizard_btn.click()
window.wizard_charuco.navigation_bar.next_wizard_step_btn.click()
# while not hasattr(window, "stereoframe"):
#     sleep(1)
# window.stereoframe.navigation_bar.calibrate_collect_btn.click()

window.show()

app.exec()