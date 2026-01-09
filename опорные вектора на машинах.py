import os
from time import time
import numpy as np
from keras.src.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 88% - Гауссова кривизна, m=n=9, ksize=3
# 90% - Средняя кривизна, m=n=9, ksize=3
# 88% - Гауссова кривизна + Средняя кривизна, m=n=9, ksize=3

def gaussian_curve(I, m=9, n=9, ksize=3):
    I_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=ksize)  # производные Собеля
    I_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=ksize)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=ksize)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=ksize)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=ksize)

    K = ((I_xx * I_yy - I_xy ** 2) / (1 + I_x ** 2 + I_y ** 2)) ** 2  # Гауссова кривизна
    # K = (I_xx * (1 + I_y ** 2) - 2 * I_xy * I_x * I_y + I_yy * (1 + I_x ** 2)) / pow((1 + I_x ** 2 + I_y ** 2), 1.5) # Средняя кривизна

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

    # Параметры для шага
    dx = height // n
    dy = width // m

    # Верхняя граница
    k += list(I[0, ::dx])

    # Нижняя граница
    k += list(I[-1, ::dx])

    # Левая граница
    k += list(I[::dy, 0])

    # Правая граница
    k += list(I[::dy, -1])

    return np.array(k)

def load_cars_vs_not_cars_dataset(data_dir='cars', img_size=(128, 128), max_samples_per_class=900):
    # Машины: 0, Не машины: 1

    images = []
    labels = []

    car_dir = os.path.join(data_dir, 'cars-1776')
    not_car_dir = os.path.join(data_dir, 'notcars-1800')

    car_files = os.listdir(car_dir)
    car_files = car_files[:max_samples_per_class]
    for i, filename in enumerate(car_files):
        try:
            img_path = os.path.join(car_dir, filename)
            img = load_img(img_path, target_size=img_size, color_mode='grayscale')
            img_array = img_to_array(img).astype('float32') / 255.0
            img_array = img_array.reshape(img_size)
            images.append(img_array)
            labels.append(0)
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")


    not_car_files = os.listdir(not_car_dir)
    not_car_files = not_car_files[:max_samples_per_class]
    for i, filename in enumerate(not_car_files):
        try:
            img_path = os.path.join(not_car_dir, filename)
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

(x_train, y_train), (x_test, y_test) = load_cars_vs_not_cars_dataset()

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

svm = SVC()

# Обучение
start_fit = time()
svm.fit(X_train_features, y_train)
fit_time = time() - start_fit
print(f"Время обучения: {fit_time:.2f}с")

# Предсказание
start_predict = time()
y_pred = svm.predict(X_test_features)
predict_time = time() - start_predict
print(f"Время предсказания: {predict_time:.2f}с")

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy:.4f}")
