# https://www.youtube.com/watch?v=Ub9lg4FWZBA

import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QTableView, QVBoxLayout,
                            QPushButton)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data


    def rowCount(self, parent: QModelIndex()):
        return len(self._data)

    def columnCount(self, parent: QModelIndex()):
        return len(max(self._data, key=len))

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        # display Data
        if role == Qt.ItemDataRole.DisplayRole:
            print('Display Role:', index.row(), index.column())
            # try except block befcause there are fewer columns for some rows
            try:
                return self._data[index.row()][index.column()]
            except IndexError:
                return ''

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            print('Edit role:', index.row(), index.column()) 
            
            if not value:
                return False
            self._data[index.row()][index.column()] = value
            self.dataChanged.emit(index,index)
        return True

    def flags(self,index):
        return super().flags(index) | Qt.ItemFlag.ItemIsEditable

    def save(self):
        print(self._data)

class MainApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 500)
        self.layout = {}
        self.layout['main'] = QVBoxLayout()
        self.setLayout(self.layout['main'])

        data_model = TableModel(data)
        self.table = QTableView()
        self.table.setModel(data_model)


        self.layout['main'].addWidget(self.table)
        self.save_btn = QPushButton("Save Data")
        self.save_btn.setMaximumSize(100, 50)

        self.layout["main"].addWidget(self.save_btn)


        self.save_btn.clicked.connect(data_model.save)

if __name__ == "__main__":
    data = [
        ['A1', "A2", "A3"],
        ['B1', "B2", "B3", "B4"],
        ['C1', "C2", "C3", "C4", "C5"]
    ]
    
    # row count
    print(len(data))
    print(max(data,key=len))

    # Column Count
    print(len(max(data,key=len)))

    app = QApplication(sys.argv)
    myApp = MainApp()

    myApp.show()

    sys.exit(app.exec())