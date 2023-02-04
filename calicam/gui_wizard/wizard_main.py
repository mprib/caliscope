import calicam.logger

logger = calicam.logger.get(__name__)

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWizard,
    QWizardPage,
    QLabel,
    QVBoxLayout,
    QRadioButton,
    QButtonGroup,
)

from calicam.gui_wizard.wiz_get_paths import WizardIntro
from calicam.gui_wizard.wizard_charuco import WizardCharuco
from calicam.session import Session
class WizardMain(QWizard):
    def __init__(self, config_path):
        super().__init__()
        self.setWindowTitle("Calibration Wizard")
        self.config_path = config_path
        self.session = Session(config_path)

        self.intro_page = WizardIntro()
        charuco_page = WizardCharuco(self.session)
        self.addPage(charuco_page) 


if __name__ == "__main__":
    from pathlib import Path
    
    repo = Path(str(Path(__file__)).split("calicam")[0], "calicam")
    config_path = Path(repo, "sessions", "high_res_session")

    app = QApplication(sys.argv)
    wizard = WizardMain(config_path)
    wizard.show()
    sys.exit(app.exec())
