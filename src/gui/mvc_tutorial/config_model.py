
from PyQt6.QtCore import Qt, QSize, QAbstractItemModel

class ConfigurationModel(QSt):

    def __init__(self, toml_path):
        super().__init__()
        self.toml_path = toml_path

