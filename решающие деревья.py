from time import time

import numpy as np
from keras.datasets import mnist
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 79% - Гауссова кривизна, m=n=7, ksize=7
# 85% - Средняя кривизна, m=n=9, ksize=7
# 79% - Гауссова кривизна и Средняя кривизна, m=n=9, ksize=7

def gaussian_curve(I, m=9, n=9):
    I_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=7) # производные Собеля
    I_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=7)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=7)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=7)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=7)

    K = ((I_xx * I_yy - I_xy ** 2) / (1 + I_x ** 2 + I_y ** 2)) ** 2  # Гауссова кривизна
    K += (I_xx * (1 + I_y ** 2) - 2 * I_xy * I_x * I_y + I_yy * (1 + I_x ** 2)) / pow((1 + I_x ** 2 + I_y ** 2), 1.5) # Средняя кривизна

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

(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
