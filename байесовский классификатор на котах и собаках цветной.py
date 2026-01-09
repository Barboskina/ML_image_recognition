import os
from time import time
import numpy as np
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 67% - Гауссоваа кривизна, m=n=9, ksize=7
# 58% - Средняя кривизна, m=n=9, ksize=9
# 67% - Гауссова кривизна + Средняя кривизна, m=n=9, ksize=7

def extract_gaussian_features(I, m=9, n=9, ksize=9):
    I_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=ksize)  # производные Собеля
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

def load_cats_vs_dogs_dataset(data_dir='PetImages', img_size=(128, 128), max_samples_per_class=900):
    # Коты: 0, Собаки: 1

    images = []
    labels = []

    cat_dir = os.path.join(data_dir, 'Cat')
    cat_files = os.listdir(cat_dir)[:max_samples_per_class]
    for filename in cat_files:
        try:
            img_path = os.path.join(cat_dir, filename)
            img = load_img(img_path, target_size=img_size, color_mode='rgb')
            img_array = img_to_array(img).astype('float32') / 255.0
            images.append(img_array)
            labels.append(0)
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")

    dog_dir = os.path.join(data_dir, 'Dog')
    dog_files = os.listdir(dog_dir)[:max_samples_per_class]
    for filename in dog_files:
        try:
            img_path = os.path.join(dog_dir, filename)
            img = load_img(img_path, target_size=img_size, color_mode='rgb')
            img_array = img_to_array(img).astype('float32') / 255.0
            images.append(img_array)
            labels.append(1)
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    print(f"\nРазмеры данных:")
    print(f"Изображения: {images.shape}")
    print(f"Метки: {labels.shape}")

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, random_state=42, stratify=labels
    )

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cats_vs_dogs_dataset()


def all_features(I): # все признаки
    features = []
    for channel in range(3):
        channel_img = I[:, :, channel]
        channel_features = extract_gaussian_features(channel_img)
        features.extend(channel_features)

    return np.array(features)

def mean_features(I): # среднее цветов
    all_features = []
    for channel in range(3):
        channel_img = I[:, :, channel]
        channel_features = extract_gaussian_features(channel_img)
        all_features.append(channel_features)

    mean_features = np.mean(all_features, axis=0)
    return mean_features

X_train_features = np.array([all_features(img) for img in x_train])
X_test_features = np.array([all_features(img) for img in x_test])

gnb = GaussianNB()

transformer = PowerTransformer(method='yeo-johnson')  # Делаем признаки более гауссовыми
X_train_features = transformer.fit_transform(X_train_features)
X_test_features = transformer.transform(X_test_features)

t_begin = time()
gnb.fit(X_train_features, y_train)
print(f"\nВремя обучения: {time() - t_begin}")

t_begin = time()
y_pred = gnb.predict(X_test_features)
print(f"\nВремя предсказания: {time() - t_begin}")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность на тестовых данных: {accuracy:.4f}")
