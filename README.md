# VIT Image Classifier

Классификатор изображений на основе Vision Transformer (ViT), обученный на датасете [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) (200 классов).

## Скачать приложение
🐧 **[Linux (x86_64)](https://huggingface.co/Krynovv/vit-tiny-imagenet/resolve/main/vit-classifier-linux.tar.gz)**

🪟 **[Windows(x64)](https://github.com/Krynovv/VIT_classify/releases/download/v1.0/ViTclassif.zip)**
### Установка
#### Linux
```bash
tar -xzf vit-classifier-linux.tar.gz
./interface/interface
```
## Демо
<img width="1415" height="820" alt="изображение" src="https://github.com/user-attachments/assets/705c541f-6369-490e-bfb0-858569d33c85" />

 
## Архитектура модели
 
| Параметр | Значение |
|---|---|
| Размер входа | 3 × 64 × 64 |
| Patch size | 4 |
| Token length | 384 |
| Num heads | 6 |
| Depth | 8 |
| Количество классов | 200 |

## Структура проекта
 
```
VIT_classify/
├── Model/
│   ├── model.py        # архитектура ViT
│   ├── attention.py
│   ├── Image2Token.py
│   └── neuralNet.py
├── checkpoints/        # здесь best_checkpoint.pt
├── interface.py        # GUI приложение (PyQt6)
├── training.py         # скрипт обучения
└── requirements.txt
```

## Чекпоинт
 
Веса модели хранятся на Hugging Face Hub:
 
🔗 [Krynovv/vit-tiny-imagenet](https://huggingface.co/Krynovv/vit-tiny-imagenet)
 
Скачайте `best_checkpoint.pt` и положите в папку `checkpoints/`:
 
```bash
mkdir checkpoints
# вручную скачайте с HuggingFace или через CLI:
huggingface-cli download Krynovv/vit-tiny-imagenet best_checkpoint.pt --local-dir checkpoints
```

## Запуск
 
```bash
python interface.py
```



