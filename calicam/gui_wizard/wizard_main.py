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

from calicam.gui_wizard.wizard_intro import WizardIntro


class WizardMain(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Wizard")

        intro_page = WizardIntro()
        self.addPage(intro_page)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    wizard = WizardMain()
    wizard.show()
    sys.exit(app.exec())
