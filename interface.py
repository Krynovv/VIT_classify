from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QSize, Qt
import sys

class MainWindow(QMainWindow):
   def __init__(self):
      super().__init__()

      self.setWindowTitle("VIT классификатор")
      self.resize(1440, 840)

      self.image_label = QLabel()
      self.image_label.setFixedSize(512, 512)
      self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

      placeholder = QPixmap(512, 512)
      placeholder.fill(Qt.GlobalColor.lightGray)
      self.image_label.setPixmap(placeholder)

      self.buttonClass = QPushButton("Классифицировать")

      self.buttonImage = QPushButton("Загрузить")
      self.buttonImage.clicked.connect(self.open_file_dialog)

      button_layout = QHBoxLayout()
      button_layout.addWidget(self.buttonClass)
      button_layout.addWidget(self.buttonImage)

      main_layout = QVBoxLayout()
      main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
      main_layout.addLayout(button_layout)


      container = QWidget()
      container.setLayout(main_layout)

      self.setCentralWidget(container)



   def open_file_dialog(self):
      file_path = QFileDialog.getOpenFileName(
         self,
         "Выберите изображение",
         ""
      )

      if file_path:
         print("Выбрано:", file_path)

app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec()