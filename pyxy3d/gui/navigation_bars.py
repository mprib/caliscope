
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QDockWidget,
    QFileDialog,
    QStackedWidget,
)

NAV_BAR_HEIGHT = 75
class NavigationBarNext(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QHBoxLayout())

        self.calibrate_collect_btn = QPushButton("&Collect Data")
        self.calibrate_collect_btn.setMaximumWidth(120)

        self.layout().setAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout().addWidget(self.calibrate_collect_btn)
        self.setMaximumHeight(NAV_BAR_HEIGHT)

class NavigationBarBackNext(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.left_box = QHBoxLayout()
        self.right_box = QHBoxLayout()

        self.layout().addLayout(self.left_box)
        self.layout().addLayout(self.right_box)

        self.back_btn = QPushButton("&Back")
        self.back_btn.setMaximumWidth(50)
        self.left_box.addWidget(self.back_btn)
        
        self.next_btn = QPushButton("&Next")
        self.next_btn.setMaximumWidth(50)
        self.right_box.addWidget(self.next_btn)
        
        self.right_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.left_box.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setMaximumHeight(NAV_BAR_HEIGHT)

        
class NavigationBarBack_deprecate(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.left_box = QHBoxLayout()
        self.right_box = QHBoxLayout()

        self.layout().addLayout(self.left_box)
        self.layout().addLayout(self.right_box)

        self.back_btn = QPushButton("&Back")
        self.back_btn.setMaximumWidth(50)
        self.left_box.addWidget(self.back_btn)
        
        self.finish_btn = QPushButton("&Finish")
        self.finish_btn.setMaximumWidth(50)
        self.right_box.addWidget(self.finish_btn)
        
        self.right_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.left_box.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setMaximumHeight(NAV_BAR_HEIGHT)

        
class NavigationBarBack(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.left_box = QHBoxLayout()
        self.right_box = QHBoxLayout()

        self.layout().addLayout(self.left_box)
        self.layout().addLayout(self.right_box)

        self.back_btn = QPushButton("&Back")
        self.back_btn.setMaximumWidth(50)
        self.left_box.addWidget(self.back_btn)
        
        # self.calibrate_collect_btn = qpushbutton("&collect data")
        # self.calibrate_collect_btn.setmaximumwidth(120)
        # self.right_box.addwidget(self.calibrate_collect_btn)
        
        self.right_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.left_box.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setMaximumHeight(NAV_BAR_HEIGHT)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    nav_bar = NavigationBarBackNext()
    nav_bar.show()
    sys.exit(app.exec())
