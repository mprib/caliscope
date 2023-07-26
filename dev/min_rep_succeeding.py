from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import QThread, Signal, Slot
from time import sleep
import random

KEY_A = 1
# KEY_A = "A"
KEY_B = "B"

class EmitterThread(QThread):
    dict_signal = Signal(dict)
       
    def run(self):
        while True:
            random_dictionary = {
                KEY_A: str(random.randint(1,10)),
                KEY_B: str(random.randint(1,10)),
                                 }

            self.dict_signal.emit(random_dictionary)
            print(f"Emitted dictionary is {random_dictionary}")
            sleep(1)

class Widget(QWidget):
    def __init__(self):
        super().__init__()

        self.label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.emitter = EmitterThread()
        self.emitter.dict_signal.connect(self.update_label)
        self.emitter.start()
       
    @Slot(dict) 
    def update_label(self, value):
        print(f"Received dictionary is {value}")
        self.label.setText(f"{KEY_A}: {value[KEY_A]}     {KEY_B}: {value[KEY_B]}")


app = QApplication([])
widget = Widget()
widget.show()

app.exec()
