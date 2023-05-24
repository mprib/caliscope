from PyQt6.QtWidgets import QDialog, QProgressBar, QVBoxLayout
from PyQt6.QtCore import Qt


class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Progress')

        self.progress_bar = QProgressBar(self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(0, 0, 300, 25)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_bar)

    def set_progress(self, value):
        self.progress_bar.setValue(value)
