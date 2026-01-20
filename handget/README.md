# HANDGET - Rock-Paper-Scissors Computer Vision AI
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-white.svg?style=flat&logo=opencv&logoColor=white)


Пет-проект для распознавания жестов "Камень-Ножницы-Бумага" в реальном времени с использованием веб-камеры. 
Проект включает в себя полный цикл ML-разработки: от сбора кастомного датасета до дообучения (Transfer Learning) модели ResNet18. В данный момент в активной разработке

## Демонстрация
![Demo GIF](assets/demo.gif) 

> Модель распознает жесты: Rock, Paper, Scissors.
> Зеленая рамка ограничивает область поиска объектов.

## Технологии
* **Python 3.12**
* **OpenCV** — захват видео, обработка изображений, отрисовка интерфейса.
* **PyTorch & Torchvision** — обучение нейросети, инференс.
* **ResNet18** — предобученная архитектура (Transfer Learning).
* **NumPy** — работа с матрицами.

## Функционал
1.  **Data Collector (`collectiondata.py`)**: 
    * Утилита для автоматического сбора датасета.
    * Позволяет быстро сохранять изображения по классам (настройка в гиперпараметрах), нажимая клавишу `s`.
    * Автоматический ресайз и сохранение структуры папок.
2.  **Training (`detection.py`, `predtrain.py`)**:
    * Использование Transfer Learning на базе ResNet18 в **predtrain.py**
    * Аугментация данных (повороты, отражения, изменение яркости) для борьбы с переобучением.
    * Валидация модели в процессе обучения.
    * Полная ручная реализация загрузчика данных и модели в **detection.py**
3.  **Real-time Inference (`detection_rl.py`)**:
    * Запуск веб-камеры.
    * Предобработка кадра (нормализация ImageNet).
    * Визуализация предсказаний от двух моделей (Bounding Box + Label + Confidence Score).

## Структура проекта
```text
├── dataset/               # Папка с собранными изображениями (в .gitignore)
├── handget_resnet.pt      # Веса дообученной модели (в .gitignore)
├── handget.pt             # Веса обученной модели (в .gitignore)
├── collectiondata.py      # Скрипт для сбора данных
├── predtrain.py           # Скрипт дообучения готовой модели
├── detection.py           # Cкрипт обучения своей модели
├── detection_rl.py        # Основной скрипт запуска
├── requirements.txt       # Зависимости
└── README.md              # Описание проекта

```


---
to-do:
- распознавание фона
- созранение по клавишам
- дистиллизация
- демонтрация
