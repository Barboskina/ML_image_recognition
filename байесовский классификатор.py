import math
from time import time

import numpy as np
from sklearn.preprocessing import PowerTransformer
from keras.datasets import mnist
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 73% - Гауссоваа кривизна, m=7,n=8, ksize=9
# 72% - Гауссова кривизна + Средняя кривизна, m=7, n=8, ksize=7
def extract_gaussian_features(I, m=7, n=8):
    I_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=7)  # производные Собеля
    I_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=7)
    I_xx = cv2.Sobel(I_x, cv2.CV_64F, 1, 0, ksize=7)
    I_yy = cv2.Sobel(I_y, cv2.CV_64F, 0, 1, ksize=7)
    I_xy = cv2.Sobel(I_x, cv2.CV_64F, 0, 1, ksize=7)

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

X_train_features = []
for img in x_train:
    X_train_features.append(extract_gaussian_features(img))
X_train_features = np.array(X_train_features)

X_test_features = []
for img in x_test:
    X_test_features.append(extract_gaussian_features(img))
X_test_features = np.array(X_test_features)

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


# Визуализация нескольких примеров
# fig, axes = plt.subplots(3, 5, figsize=(10, 10))
# for i in range(15):
#     ax = axes[i // 5, i % 5]
#     ax.imshow(x_test[i], cmap='gray')
#     ax.set_title(f'True: {y_test[i]}, Pred: {y_pred[i]}')
#     ax.axis('off')
# plt.tight_layout()
# plt.show()