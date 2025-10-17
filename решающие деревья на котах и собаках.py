import os
from time import time
import numpy as np
from keras.src.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 57% - Гауссова кривизна, m=n=9, ksize=7
# 56% - Средняя кривизна, m=n=9, ksize=3
# 57% - Гауссова кривизна и Средняя кривизна, m=n=9, ksize=7

def gaussian_curve(I, m=9, n=9, ksize=7):
    I_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=ksize) # производные Собеля
    I_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=ksize)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=ksize)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=ksize)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=ksize)

    K = ((I_xx * I_yy - I_xy ** 2) / (1 + I_x ** 2 + I_y ** 2)) ** 2  # Гауссова кривизна
    #K = (I_xx * (1 + I_y ** 2) - 2 * I_xy * I_x * I_y + I_yy * (1 + I_x ** 2)) / pow((1 + I_x ** 2 + I_y ** 2), 1.5) # Средняя кривизна

    height, width = I.shape
    block_height = height // m  # находим дельта i,j
    block_width = width // n

    k = []

    for i in range(m):
        for j in range(n):
            start_i = i * block_height
            end_i = (i + 1) * block_height
            start_j = j * block_width
            end_j = (j + 1) * block_width

            block = K[start_i:end_i, start_j:end_j]
            result = np.sum(block) / (m * n)
            k.append(result)

    return np.array(k)

def load_cats_vs_dogs_dataset(data_dir='PetImages', img_size=(128, 128), max_samples_per_class=900):
    # Коты: 0, Собаки: 1

    images = []
    labels = []

    cat_dir = os.path.join(data_dir, 'Cat')
    dog_dir = os.path.join(data_dir, 'Dog')

    cat_files = os.listdir(cat_dir)
    cat_files = cat_files[:max_samples_per_class]
    for i, filename in enumerate(cat_files):
        try:
            img_path = os.path.join(cat_dir, filename)
            img = load_img(img_path, target_size=img_size, color_mode='grayscale')
            img_array = img_to_array(img).astype('float32') / 255.0
            img_array = img_array.reshape(img_size)
            images.append(img_array)
            labels.append(0)
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")


    dog_files = os.listdir(dog_dir)
    dog_files = dog_files[:max_samples_per_class]
    for i, filename in enumerate(dog_files):
        try:
            img_path = os.path.join(dog_dir, filename)
            img = load_img(img_path, target_size=img_size, color_mode='grayscale')
            img_array = img_to_array(img).astype('float32') / 255.0
            img_array = img_array.reshape(img_size)
            images.append(img_array)
            labels.append(1)
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, random_state=42, stratify=labels
    )

    print(f"\nРазмеры данных:")
    print(f"Тренировочные данные: {x_train.shape}")
    print(f"Тестовые данные: {x_test.shape}")

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cats_vs_dogs_dataset()

# Тренировочные данные
X_train_features = []
for i, image in enumerate(x_train):
    features = gaussian_curve(image)
    X_train_features.append(features)
X_train_features = np.array(X_train_features)

# Тестовые данные
X_test_features = []
for i, image in enumerate(x_test):
    features = gaussian_curve(image)
    X_test_features.append(features)
X_test_features = np.array(X_test_features)

classifier = DecisionTreeClassifier()
t_begin = time()
classifier.fit(X_train_features, y_train)
print(f"Время обучения: {time() - t_begin:.2f} сек")

t_begin = time()
y_pred = classifier.predict(X_test_features)
print(f"Время предсказания: {time() - t_begin:.2f} сек")

accuracy = accuracy_score(y_test, y_pred)

print(f"Точность: {accuracy}")
