import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# Import the global handler instance directly
from caliscope.logger import qt_handler_instance

logger = logging.getLogger(__name__)


class LogWidget(QWidget):
    def __init__(self, message: str = None):
        super(LogWidget, self).__init__()
        self.setWindowTitle(message)
        self._console = LogMessageViewer(self)

        self.setWindowFlags(Qt.WindowType.WindowTitleHint)

        layout = QVBoxLayout()
        layout.addWidget(self._console)

        self.setLayout(layout)

        qt_handler_instance.emitter.message_written.connect(self._console.appendLogMessage)


class LogMessageViewer(QTextBrowser):
    def __init__(self, parent=None):
        super(LogMessageViewer, self).__init__(parent)
        self.setReadOnly(True)
        self.setEnabled(True)
        self.verticalScrollBar().setVisible(True)

    @Slot(str)
    def appendLogMessage(self, msg: str):
        """Appends a message to the text browser and scrolls to the bottom."""
        # Ensure we scroll to the bottom to see the latest message
        verScrollBar = self.verticalScrollBar()
        scrollIsAtEnd = verScrollBar.value() >= (verScrollBar.maximum() - 10)

        self.insertPlainText(msg)

        if scrollIsAtEnd:
            verScrollBar.setValue(verScrollBar.maximum())
