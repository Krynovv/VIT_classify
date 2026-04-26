from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QSize, Qt
import sys
import torch
from torchvision import transforms
from PIL import Image
from Model.model import VIT_Model
from datasets import load_dataset

model = VIT_Model(
   img_size=(3, 64, 64),
   patch_size=4,
   token_len=384,
   preds=200,
   num_heads=6,
   Encoding_hidden_chan_mul=4.,
   depth=8
)

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load('checkpoints/best_checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(device)

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


ds = load_dataset("zh-plus/tiny-imagenet")
features = ds["train"].features["label"]

CLASSES = features.names  

LABEL_MAP = {
    "n01443537": "goldfish", "n01629819": "European fire salamander",
    "n01641577": "bullfrog", "n01644900": "tailed frog",
    "n01698640": "American alligator", "n01742172": "boa constrictor",
    "n01768244": "trilobite", "n01770393": "scorpion",
    "n01774384": "black widow", "n01774750": "tarantula",
    "n01784675": "centipede", "n01855672": "goose",
    "n01882714": "koala", "n01910747": "jellyfish",
    "n01917289": "brain coral", "n01944390": "snail",
    "n01945685": "slug", "n01950731": "sea slug",
    "n01983481": "American lobster", "n01984695": "spiny lobster",
    "n02002724": "black stork", "n02056570": "king penguin",
    "n02058221": "albatross", "n02074367": "dugong",
    "n02085620": "Chihuahua", "n02094433": "Yorkshire terrier",
    "n02099601": "golden retriever", "n02099712": "Labrador retriever",
    "n02106662": "German shepherd", "n02113799": "standard poodle",
    "n02123045": "tabby cat", "n02123159": "Persian cat",
    "n02124075": "Egyptian cat", "n02125311": "cougar",
    "n02129165": "lion", "n02132136": "brown bear",
    "n02133161": "ice bear", "n02488291": "langur",
    "n02165456": "ladybug", "n02190166": "fly",
    "n02206856": "bee", "n02226429": "grasshopper",
    "n02231487": "walking stick", "n02233338": "cockroach",
    "n02236044": "mantis", "n02268443": "dragonfly",
    "n02279972": "monarch butterfly", "n02281406": "cabbage butterfly",
    "n02281787": "sulphur butterfly", "n02317335": "sea cucumber",
    "n02364673": "guinea pig", "n02395406": "hog",
    "n02410509": "ox", "n02415577": "bighorn sheep",
    "n02423022": "gazelle", "n02437312": "Arabian camel",
    "n02480495": "orangutan", "n02481823": "chimpanzee",
    "n02486261": "baboon", "n02504458": "African elephant",
    "n02509815": "lesser panda", "n02666196": "abacus",
    "n02669723": "academic gown", "n02699494": "altar",
    "n02730930": "apron", "n02769748": "backpack",
    "n02788148": "bannister", "n02791270": "barbershop",
    "n02793495": "barn", "n02795169": "barrel",
    "n02802426": "basketball", "n02808440": "bathtub",
    "n02814533": "beach wagon", "n02814860": "beacon",
    "n02815834": "beaker", "n02823428": "beer bottle",
    "n02837789": "bikini", "n02841315": "binoculars",
    "n02843684": "birdhouse", "n02883205": "bow tie",
    "n02892201": "brass", "n02906734": "broom",
    "n02909870": "bucket", "n02917067": "bullet train",
    "n02927161": "butcher shop", "n02948072": "candle",
    "n02950826": "cannon", "n02963159": "cardigan",
    "n02977058": "cash machine", "n02988304": "CD player",
    "n02999410": "chain", "n03014705": "chest",
    "n03026506": "Christmas stocking", "n03042490": "cliff dwelling",
    "n03085013": "computer keyboard", "n03089624": "confectionery",
    "n03100240": "convertible", "n03126707": "crane",
    "n03160309": "dam", "n03179701": "desk",
    "n03201208": "dining table", "n03250847": "drumstick",
    "n03255030": "dumbbell", "n03355925": "flagpole",
    "n03388043": "fly fishing vest", "n03393912": "freight car",
    "n03400231": "frying pan", "n03404251": "fur coat",
    "n03424325": "gasmask", "n03444034": "go-kart",
    "n03447447": "gondola", "n03544143": "hourglass",
    "n03584254": "iPod", "n03599486": "jinrikisha",
    "n03617480": "kimono", "n03637318": "lampshade",
    "n03649909": "lawn mower", "n03662601": "lifeboat",
    "n03670208": "limousine", "n03706229": "magnetic compass",
    "n03733131": "maypole", "n03763968": "military uniform",
    "n03770439": "miniskirt", "n03796401": "moving van",
    "n03804744": "nail", "n03814639": "neck brace",
    "n03837869": "obelisk", "n03838899": "oboe",
    "n03854065": "organ", "n03891332": "parking meter",
    "n03902125": "pay phone", "n03930313": "pickup truck",
    "n03937543": "pill bottle", "n03970156": "plunger",
    "n03976657": "pole", "n03977966": "police van",
    "n03980874": "poncho", "n03983396": "pop bottle",
    "n03992509": "pot", "n04008634": "potter's wheel",
    "n04023962": "power drill", "n04099969": "rocking chair",
    "n04118538": "rugby ball", "n04133789": "sandal",
    "n04146614": "school bus", "n04149813": "scoreboard",
    "n04179913": "sewing machine", "n04251144": "snorkel",
    "n04254777": "sock", "n04259630": "sombrero",
    "n04265275": "space heater", "n04275548": "spider web",
    "n04285008": "sports car", "n04311004": "steel arch bridge",
    "n04328186": "stopwatch", "n04356056": "sunglasses",
    "n04366367": "suspension bridge", "n04371430": "swimming trunks",
    "n04376876": "syringe", "n04398044": "teapot",
    "n04399382": "teddy bear", "n04417672": "thatch",
    "n04456115": "torch", "n04462240": "tractor",
    "n04465501": "triumphal arch", "n04467665": "trolleybus",
    "n04479046": "turnstile", "n04485082": "umbrella",
    "n04501370": "vestment", "n04507155": "viaduct",
    "n04532106": "volleyball", "n04532670": "water jug",
    "n04540053": "water tower", "n04560804": "wok",
    "n04562935": "wooden spoon", "n04596742": "worm fence",
    "n04598010": "wreck", "n06596364": "yawl",
    "n07056507": "yo-yo", "n07583066": "mushroom",
    "n07614500": "ice cream", "n07615774": "cliff",
    "n07695742": "pretzel", "n07711569": "mashed potato",
    "n07715103": "cauliflower", "n07720875": "bell pepper",
    "n07734744": "mushroom", "n07747607": "orange",
    "n07749582": "lemon", "n07753592": "banana",
    "n07760859": "pomegranate", "n07768694": "pineapple",
    "n07871810": "meat loaf", "n07873807": "pizza",
    "n07875152": "potpie", "n07920052": "espresso",
    "n07930864": "cup", "n09193705": "glacier",
    "n09246464": "cliff", "n09256479": "coral reef",
    "n09332890": "lakeside", "n09428293": "seashore",
    "n09468604": "valley", "n09472597": "volcano",
    "n09835506": "ballplayer", "n10565667": "scuba diver",
    "n11879895": "rapeseed", "n11939491": "daisy",
    "n12057211": "yellow lady's slipper", "n12144580": "corn",
    "n12267677": "acorn",
}


class MainWindow(QMainWindow):
   def __init__(self):
      super().__init__()
      self.current_image_path = None

      self.setWindowTitle("VIT классификатор")
      self.resize(1440, 840)

      self.image_label = QLabel()
      self.image_label.setFixedSize(512, 512)
      self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
      self.image_label.setPixmap(self._make_placeholder())
      self.image_label.setObjectName("image_label")

      self.result_label = QLabel("Результат: ")
      self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

      self.buttonClass = QPushButton("Классифицировать")
      self.buttonClass.clicked.connect(self.classify_image)
      self.buttonClass.setObjectName("buttonClass")
      self.buttonClass.setFixedHeight(50)

      self.buttonImage = QPushButton("Загрузить")
      self.buttonImage.clicked.connect(self.open_file_dialog)
      self.buttonImage.setObjectName("buttonImage")
      self.buttonImage.setText("⤓")
      self.buttonImage.setFixedSize(50, 50)

      card_layout = QVBoxLayout()
      card_layout.addWidget(self.image_label)

      card = QWidget()
      card.setLayout(card_layout)
      card.setStyleSheet("background-color: #3a3a3a; border-radius: 8px; padding: 8px;")

      button_layout = QHBoxLayout()
      button_layout.setSpacing(8)
      button_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
      button_layout.addStretch()
      button_layout.addWidget(self.buttonClass, alignment=Qt.AlignmentFlag.AlignVCenter)
      button_layout.addWidget(self.buttonImage, alignment=Qt.AlignmentFlag.AlignVCenter)
      button_layout.addStretch()
      button_layout.setContentsMargins(0, 0, 5, 0)

      main_layout = QVBoxLayout()
      main_layout.setContentsMargins(40, 40, 40, 40)
      main_layout.addWidget(card, alignment=Qt.AlignmentFlag.AlignCenter)
      main_layout.addWidget(self.result_label)
      main_layout.addLayout(button_layout)

      container = QWidget()
      container.setLayout(main_layout)

      self.setCentralWidget(container)

   def _make_placeholder(self):
      from PyQt6.QtGui import QPainter, QColor, QPen, QPolygon
      from PyQt6.QtCore import QPoint
      px = QPixmap(512, 512)
      px.fill(QColor("#2a2a2a"))
      painter = QPainter(px)
      painter.setRenderHint(QPainter.RenderHint.Antialiasing)

      # icon color
      color = QColor("#555555")
      painter.setPen(QPen(color, 3))
      painter.setBrush(color)

      cx, cy = 256, 256
      w, h = 90, 66

      # mountain shape
      pts = QPolygon([
         QPoint(cx - w, cy + h // 2),
         QPoint(cx - w // 4, cy - h // 2),
         QPoint(cx + w // 4, cy),
         QPoint(cx + w // 2, cy - h // 3),
         QPoint(cx + w, cy + h // 2),
      ])
      painter.drawPolygon(pts)

      # sun circle
      painter.drawEllipse(cx - w + 14, cy - h // 2 - 2, 22, 22)

      painter.end()
      return px

   def _make_error_placeholder(self):
      from PyQt6.QtGui import QPainter, QColor, QPen, QFont
      px = QPixmap(512, 512)
      px.fill(QColor("#2a2a2a"))
      painter = QPainter(px)
      painter.setRenderHint(QPainter.RenderHint.Antialiasing)

      cx, cy = 256, 220

      # red circle
      painter.setBrush(QColor("#c0392b"))
      painter.setPen(QPen(QColor("#c0392b"), 0))
      painter.drawEllipse(cx - 36, cy - 36, 72, 72)

      # white exclamation mark
      painter.setPen(QPen(QColor("#ffffff"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
      painter.drawLine(cx, cy - 18, cx, cy + 8)
      painter.setBrush(QColor("#ffffff"))
      painter.setPen(QPen(QColor("#ffffff"), 0))
      painter.drawEllipse(cx - 4, cy + 16, 8, 8)

      # text below
      painter.setPen(QColor("#aaaaaa"))
      font = QFont("Segoe UI", 11)
      painter.setFont(font)
      painter.drawText(0, cy + 70, 512, 30, Qt.AlignmentFlag.AlignHCenter, "Некорректный тип данных")

      painter.end()
      return px

   def classify_image(self):
      if self.current_image_path is None:
         self.result_label.setText("Загрузите изображение")
         return

      img = Image.open(self.current_image_path)
      tensor = transform(img).unsqueeze(0).to(device)

      with torch.no_grad():
         output = model(tensor)
         predicted = output.argmax(dim=1).item()
         confidence = torch.softmax(output, dim=1)[0][predicted].item()

         wnid = CLASSES[predicted]
         class_name = LABEL_MAP.get(wnid, wnid)

         self.result_label.setText(f"Класс: {class_name} | Уверенность: {confidence:.1%}")

   def open_file_dialog(self):
      file_path, _ = QFileDialog.getOpenFileName(
         self,
         "Выберите изображение",
         "",
         "Изображение (*.png *.jpg *.jpeg)"
      )

      if file_path:
         import os
         ext = os.path.splitext(file_path)[1].lower()
         if ext not in (".png", ".jpg", ".jpeg"):
            self.current_image_path = None
            self.image_label.setPixmap(self._make_error_placeholder())
            self.result_label.setText("")
            return
         self.current_image_path = file_path
         pixmap = QPixmap(file_path).scaled(
            512, 512,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
         )
         self.image_label.setPixmap(pixmap)
         self.result_label.setText("Результат: ")

app = QApplication(sys.argv)
app.setStyleSheet("""
                  QMainWindow {
                     background-color: #1E1E1E;
                  }

                  QWidget {
                     background-color: #1E1E1E;
                     color: #ffffff;
                     font-family: 'Segoe UI';
                     font-size: 14px;
                  }

                  QLabel#image_label {
                     background-color: #3a3a3a;
                     border-radius: 8px;
                  }

                  QLabel#image_title {
                     background-color: #3a3a3a;
                     border-radius: 6px;
                     padding: 4px 12px;
                     color: #cccccc;
                     font-size: 12px;
                  }

                  QPushButton#buttonClass {
                     background-color: #3a3a3a;
                     color: #ffffff;
                     border: none;
                     border-radius: 12px;
                     font-size: 14px;
                     padding: 10px 28px;
                     min-height: 44px;
                  }

                  QPushButton#buttonClass:hover {
                     background-color: #4a4a4a;
                  }

                  QPushButton#buttonImage {
                     background-color: #7c4daa;
                     border-radius: 12px;
                     border: none;
                     padding: 10px 14px;
                     min-width: 44px;
                     min-height: 44px;
                     color: #ffffff;
                     font-size: 20px;
                  }

                  QPushButton#buttonImage:hover {
                     background-color: #9c6dca;
                  }
""")
window = MainWindow()
window.show()

app.exec()
