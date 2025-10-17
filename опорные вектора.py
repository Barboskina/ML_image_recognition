from time import time

import numpy as np
from keras.datasets import mnist
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 61% - Гауссова кривизна, m=n=7, ksize=5
# 90% - Средняя кривизна, m=n=7, ksize=5
def gaussian_curve(I, m=7, n=7):
    I_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)  # производные Собеля
    I_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=5)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=5)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=5)

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
