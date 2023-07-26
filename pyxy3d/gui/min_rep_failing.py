from PySide6.QtWidgets import ( QApplication, QWidget, QLabel, QVBoxLayout,)
from PySide6.QtCore import Signal,Slot, QThread
import sys

from time import perf_counter, sleep

import random
 
class MyWidget(QWidget):
     
    def __init__(self):

        super(MyWidget, self).__init__()

        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.label)

        self.setLayout(layout)

        self.emitter = EmitterThread()
        self.emitter.dict_signal.connect(self.update_label)
        self.emitter.start()
        
    @Slot(dict) 
    def update_label(self, data):
        "Unravel dropped fps dictionary to a more readable string"
        print(f"Just received {data}")
        self.label.setText(f"{data['1']}")
         
class EmitterThread(QThread):
    dict_signal = Signal(dict)
       
    def run(self):
        while True:

            random_dictionary = {1:str(random.randint(1,10))}
            self.dict_signal.emit(random_dictionary)
            print(f"Emit dictionary: {random_dictionary}")

            sleep(1)

App = QApplication([])
widget = MyWidget()
widget.show()
App.exec()
# sys.exit(App.exec())