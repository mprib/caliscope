from PySide6.QtWidgets import ( QApplication, QWidget, QLabel, QVBoxLayout,)
from PySide6.QtCore import Signal,Slot, QThread
from time import sleep
import random
 
class MyWidget(QWidget):
     
    def __init__(self):
        super().__init__()

        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.label)

        self.setLayout(layout)

        self.emitter = EmitterThread()
        self.emitter.my_signal.connect(self.update_label)
        self.emitter.start()
        
    @Slot(dict) 
    def update_label(self, data):
        "Unravel dropped fps dictionary to a more readable string"
        print(f"Just received {data}")
        self.label.setText(f"{data['1']}")
         
class EmitterThread(QThread):
    my_signal = Signal(dict)
       
    def run(self):
        while True:

            emitted_dictionary = {"1":str(random.randint(1,10))}
            self.my_signal.emit(emitted_dictionary)
            print(f"Emit dictionary: {emitted_dictionary}")

            sleep(1)

App = QApplication([])
widget = MyWidget()
widget.show()
App.exec()
# sys.exit(App.exec())