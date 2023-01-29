# Built following the tutorials that begin here:
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/
import logging

LOG_FILE = "log\camera_table.log"
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
LOG_FORMAT = " %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

logging.basicConfig(filename=LOG_FILE, filemode="w", format=LOG_FORMAT, level=LOG_LEVEL)

import sys
from pathlib import Path

from numpy import char
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calicam.gui.camera_config.camera_config_dialogue import CameraConfigDialog
from calicam.gui.charuco_builder import CharucoBuilder
from calicam.session import Session


class CameraTable(QWidget):
    def __init__(self, session):
        super().__init__()

        vbox = QVBoxLayout()

        self.session = session
        self.table = QTableWidget()
        # make table read only
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.setLayout(vbox)
        vbox.addWidget(self.table)

        self.update_data()
        self.table.verticalHeader().setVisible(False)
        self.table.resizeColumnsToContents()

    def update_data(self):
        """Builds a list-of-lists table structure from the config file and
        then writes this to a table"""

        # start fresh each time
        self.data = []
        # columns to import from camera configs
        self._headers = ["port", "resolution", "error", "grid_count"]

        for key, params in self.session.config.items():
            if "cam" in key:
                print(f"Found {key}")
                if "error" in params.keys():
                    pass
                else:
                    params["error"] = None
                    params["grid_count"] = 0
                params = {k: params[k] for k in self._headers}
                res = params["resolution"]
                params["resolution"] = f"{res[0]} x {res[1]}"
                self.data.append(params)

        logging.debug(f"Updating cam data to {self.data}")

        row_count = len(self.data)
        column_count = len(self._headers)

        self.table.setRowCount(row_count)
        self.table.setColumnCount(column_count)
        self.table.setHorizontalHeaderLabels(self._headers)

        for row in range(row_count):
            for column in range(column_count):
                item = list(self.data[row].values())[column]
                print(item)
                self.table.setItem(row, column, QTableWidgetItem(str(item)))


if __name__ == "__main__":
    repo = Path(__file__).parent.parent.parent.parent
    config_path = Path(repo, "sessions", "high_res_session")
    
    session = Session(config_path)
    print(session.config)
    app = QApplication(sys.argv)
    window = CameraTable(session)
    window.show()
    app.exec()
