from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import QThread, Signal, Slot
from time import sleep
import random



class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.label_A = QLabel('A: N/A')
        self.label_B = QLabel('B: N/A')

        layout = QVBoxLayout()
        layout.addWidget(self.label_A)
        layout.addWidget(self.label_B)

        self.setLayout(layout)

        self.my_thread = MyThread()
        self.my_thread.my_signal.connect(self.update_labels)
        self.my_thread.start()

    @Slot(dict)
    def update_labels(self, data):
        print(f"receiving {data}")
        self.label_A.setText(f'A: {data[1]}')
        self.label_B.setText(f'B: {data["2"]}')


class MyThread(QThread):
    my_signal = Signal(dict)

    def run(self):
        while True:
            # Emitting a dictionary with random values
            emitted_dictionary = {1: random.randint(1, 10), '2': random.randint(1, 20)}
            self.my_signal.emit(emitted_dictionary)
            print(f"Emitting {emitted_dictionary}")
            print(emitted_dictionary[1])

            sleep(1)

app = QApplication([])
widget = MyWidget()
widget.show()

app.exec()
