from PySide6.QtWidgets import QLabel, QDialog, QProgressBar, QVBoxLayout
from PySide6.QtCore import Qt


class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Landmark Tracking and Triangulation')

        self.display_text = QLabel()
        self.progress_bar = QProgressBar(self)
        # self.progress_bar.setGeometry(0, 0, 300, 25)
        # self.progress_bar.setMaximum(100)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.progress_bar.setFormat("This is a test")

        layout = QVBoxLayout(self)
        layout.addWidget(self.display_text)
        layout.addWidget(self.progress_bar)

    def update(self, data:dict):
        if "close" in data.keys():
            self.hide()
        else:
            self.display_text.setText(data["stage"])
            self.progress_bar.setValue(data["percent"])


