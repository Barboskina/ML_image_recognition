from time import time

import numpy as np
from keras.datasets import mnist
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 60% - Гауссова кривизна, m=n=7, ksize=7
# 85% - Средняя кривизна, m=n=7, ksize=5, k=9
def gaussian_curve(I, m=7, n=7):
    I_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)  # производные Собеля
    I_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=5)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=5)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=5)

    #K = ((I_xx * I_yy - I_xy ** 2) / (1 + I_x ** 2 + I_y ** 2)) ** 2  # Гауссова кривизна
    K = (I_xx * (1 + I_y ** 2) - 2 * I_xy * I_x * I_y + I_yy * (1 + I_x ** 2)) / pow((1 + I_x ** 2 + I_y ** 2), 1.5) # Средняя кривизна

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
for image in x_train:
    features = gaussian_curve(image)
    X_train_features.append(features)
X_train_features = np.array(X_train_features)

# Тестовые данные
X_test_features = []
for image in x_test:
    features = gaussian_curve(image)
    X_test_features.append(features)

X_test_features = np.array(X_test_features)

t_begin = time()
# Подбор оптимального k
best_accuracy = 0
best_k = 1
for k in range(1, 18):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_features, y_train)
    y_pred = knn.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"k = {k}, Accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"\nЛучшее k: {best_k}, Лучшая точность: {best_accuracy:.4f}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_features, y_train)
print(f"\nВремя обучения: {time() - t_begin}")

t_begin = time()
y_pred = knn_final.predict(X_test_features)
print(f"\nВремя предсказания: {time() - t_begin}")

final_accuracy = accuracy_score(y_test, y_pred)
print(f"\nФинальная точность на тестовых данных: {final_accuracy:.4f}")

# Визуализация нескольких примеров
# fig, axes = plt.subplots(3, 5, figsize=(10, 10))
# for i in range(15):
#     ax = axes[i // 5, i % 5]
#     ax.imshow(x_test[i], cmap='gray')
#     ax.set_title(f'True: {y_test[i]}, Pred: {y_pred[i]}')
#     ax.axis('off')
#
# plt.tight_layout()
# plt.show()
