import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
from tensorflow.keras.preprocessing import image
import numpy as np

# Завантажуємо модель
model = tf.keras.models.load_model('best_classifier_model.keras')

# Шлях до папки з зображеннями
image_dir = '/Users/maxim/PycharmProjects/ImageClassification+NER/eval_dataset'

# Отримаємо список зображень
images = os.listdir(image_dir)

# Ініціалізація списку для зображень та передбачень
image_data = []
predictions = []

# Завантажуємо та передбачаємо для кожного зображення
for img_name in images:
    img_path = os.path.join(image_dir, img_name)

    # Завантажуємо зображення
    img = image.load_img(img_path, target_size=(224, 224))  # Задайте правильний розмір відповідно до вашої моделі
    img_array = image.img_to_array(img) / 255.0  # Нормалізація
    img_array = np.expand_dims(img_array, axis=0)  # Додаємо вимір для батчу

    # Передбачення
    prediction = model.predict(img_array)
    predictions.append(prediction)

    # Зберігаємо зображення для візуалізації
    image_data.append(img)

# Визначаємо кількість зображень в ряду
images_per_row = 2

# Рахуємо кількість рядків
rows = len(images) // images_per_row + (1 if len(images) % images_per_row != 0 else 0)

# Візуалізація результатів
fig, axes = plt.subplots(rows, images_per_row, figsize=(15, 5 * rows))

# Якщо лише один рядок
if rows == 1:
    axes = [axes]

# Виводимо зображення по два в рядку
for i, (ax, img, pred, img_name) in enumerate(zip(axes.flat, image_data, predictions, images)):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'{img_name}\nPrediction: {np.argmax(pred)}')  # Показуємо передбачення

# Якщо є порожні осі (для непарної кількості зображень)
for i in range(len(images), len(axes.flat)):
    axes.flat[i].axis('off')

plt.tight_layout()
plt.show()
