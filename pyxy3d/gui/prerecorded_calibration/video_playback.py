from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        
        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.on_play)
        
        self.open_button = QPushButton("Open", self)
        self.open_button.clicked.connect(self.on_open)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_widget)
        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.open_button)
        
        self.setLayout(self.layout)
    
    def on_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")
    
    def on_open(self):
        file_dialog = QFileDialog(self)
        file_dialog.setMimeTypeFilters(["video/mp4"])
        file_dialog.setViewMode(QFileDialog.List)
        
        if file_dialog.exec():
            video_file = file_dialog.selectedFiles()[0]
            self.media_player.setMedia(video_file)
            self.media_player.play()
            self.play_button.setText("Pause")
            
if __name__ == "__main__":
    app = QApplication([])
    player = VideoPlayer()
    player.show()
    app.exec()
