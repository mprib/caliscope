from PyQt6.QtWidgets import QMainWindow, QStackedLayout



class MainWindow(QMainWindow):
        def __init__(self):
            super(MainWindow, self).__init__()
             
            
            self.setLayout(QStackedLayout)

            