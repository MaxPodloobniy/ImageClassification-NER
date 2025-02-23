import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))


# Отримуємо список усіх файлів
all_files = []
for class_name in os.listdir('/Users/maxim/PycharmProjects/ImageClassification+NER/dataset'):
    if class_name.lower() != "hippopotamus":  # Виключаємо гіпопотамів
        class_path = os.path.join('/Users/maxim/PycharmProjects/ImageClassification+NER/dataset', class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                all_files.append((os.path.join(class_path, filename), class_name))

# Створюємо DataFrame
df = pd.DataFrame(all_files, columns=["filename", "class"])

# Ділимо на train і val, щоб зберегти всі класи в обох частинах
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=42)

# Тепер створюємо генератори без validation_split
train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=24,
    shuffle=True,
    seed=42,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=2,
    shuffle=True,
    seed=42,
    class_mode="categorical"
)


print(f'Train gen classes:{train_generator.class_indices}')
print(f'Test gen classes:{valid_generator.class_indices}')

print(f'Minimal value on image: {tf.reduce_min(train_generator[0][0])}')
print(f'Maximal value on image: {tf.reduce_max(train_generator[0][0])}')


x_batch, y_batch = next(train_generator)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for img, lbl, ax in zip(x_batch[:9], y_batch[:9], axes):
    ax.imshow(img)
    ax.set_title(f"Class: {np.argmax(lbl)}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Завантажуємо модель без верхнього шару (fully connected layers)
base_model = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3),
    include_top=False,  # we dont need a pre-trained top layer (output layer)
    weights='imagenet'
)

for layer in base_model.layers[:15]:
    layer.trainable = True
for layer in base_model.layers[15:]:
    layer.trainable = False

# Створюємо нову модель
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Компільовуємо модель
model.compile(optimizer=Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Колбек для збереження найкращої моделі
checkpoint_callback = ModelCheckpoint(
    "best_classifier_model.keras",  # Назва файлу для збереження
    monitor="val_accuracy",  # Відстежуємо точність на валідації
    save_best_only=True,  # Зберігаємо тільки найкращу версію моделі
    mode="max",  # Найвищий показник val_accuracy – найкраща модель
    verbose=1
)

# Колбек для раннього зупинення (щоб не перенавчатися)
early_stopping_callback = EarlyStopping(
    patience=4,  # Скільки епох чекати без покращення
    restore_best_weights=True  # Повертає найкращі ваги після завершення
)

# Зниження фактора навчання для більш тонкого навчання моделі
learning_rate_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    mode='min'
)

model.fit(
    train_generator,
    epochs=100,
    validation_data=valid_generator,
    callbacks=[checkpoint_callback, early_stopping_callback, learning_rate_reduce]
)
