import numpy as np
from collections import Counter


def euclid_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_predict(x_train, y_train, x_test, k=3):
    distance = []
    for i in range(len(x_train)):
        dist = euclid_distance(x_test, x_train[i])
        distance.append((dist, y_train[i]))
    distance.sort(key=lambda x: x[0])
    neighbors = distance[:k]
    output = [neighbor[1] for neighbor in neighbors]
    prediction = sum(output) / len(output)
    return prediction


def accuracy(x_train, y_train, x_test, y_test, k=3):
    mse = 0.0
    for i in range(len(x_test)):
        prediction = knn_predict(x_train, y_train, x_test[i], k)
        mse += (prediction - y_test[i]) ** 2
    return mse / len(x_test)


x_train = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]])
y_train = np.array([5.2, 5.5, 5.0, 7.1, 7.8, 8.0])
x_test = np.array([[1, 1], [8, 9]])
y_test = np.array([5.0, 8.2])


k = 3
mse_score = accuracy(x_train, y_train, x_test, y_test, k)
print(f"Mean Squared Error: {mse_score:.2f}")
for i in range(len(x_test)):
    print(f"Predicted value for {x_test[i]}: {knn_predict(x_train, y_train, x_test[i], k):.2f}")