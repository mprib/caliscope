# Built following the tutorials that begin here: 
# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/

from re import L
import sys

from PyQt6.QtCore import Qt, QAbstractTableModel
from PyQt6.QtGui import QPalette, QColor, QIcon
from PyQt6.QtWidgets import ( QVBoxLayout, QHBoxLayout, QLabel, QMainWindow, 
                            QPushButton, QTabWidget, QWidget,QGroupBox, QTableWidgetItem,
                            QScrollArea, QApplication, QTableWidget, QTableView)

from pathlib import Path
from numpy import char

from qtpy import QT_VERSION

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.gui.charuco_builder import CharucoBuilder
from src.gui.camera_config_dialogue import CameraConfigDialog
from src.session import Session


# import sys
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtCore import Qt

class DictionaryTableModel(QAbstractTableModel):
    def __init__(self, data, headers):
        super(DictionaryTableModel, self).__init__()
        self._data = data
        self._headers = headers

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            # Look up the key by header index.
            column = index.column()
            column_key = self._headers[column]
            return self._data[index.row()][column_key]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The length of our headers.
        return len(self._headers)

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._headers[section])

            if orientation == Qt.Orientation.Vertical:
                return str(section)

class CameraTable(QWidget):
    def __init__(self,session):
        super().__init__()

        vbox = QVBoxLayout()

        self.session = session
        self.table = QTableWidget()
       
        self.setLayout(vbox)
        vbox.addWidget(self.table)   

        self.update_data()        
        self.table.verticalHeader().setVisible(False)
        self.table.resizeColumnsToContents()


    def update_data(self):
        """ Builds a list-of-lists table structure from the config file and 
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
                # print(params)
                print(params)
                params = {k: params[k] for k in self._headers}
                res = params["resolution"]
                params["resolution"] = f"{res[0]} x {res[1]}"
                self.data.append(params)

        print(f"Updating cam data to {self.data}")

        row_count = len(self.data)
        column_count = len(self._headers)

        self.table.setRowCount(row_count)
        self.table.setColumnCount(column_count)
        self.table.setHorizontalHeaderLabels(self._headers)
        # alternate = ["1", "2", "3", "$"]
        # self.table.setHorizontalHeaderLabels(alternate)

        for row in range(row_count):
            for column in range(column_count):
                item = list(self.data[row].values())[column]
                print(item)
                self.table.setItem(row,column, QTableWidgetItem(str(item)))

if __name__ == "__main__":

    session = Session(r'C:\Users\Mac Prible\repos\learn-opencv\test_session')
    print(session.config)
    app=QApplication(sys.argv)
    window=CameraTable(session)
    window.show()
    app.exec()
