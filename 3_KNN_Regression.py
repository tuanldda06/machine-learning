import numpy as np
import matplotlib.pyplot as plt

def euclid_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_regression(x_train, y_train, x_test, k=4):
    distance = []
    for i in range(len(x_train)):
        dist = euclid_distance(x_train[i], x_test)
        distance.append((dist, y_train[i]))
    distance.sort(key=lambda x: x[0])   # sắp xếp theo khoảng cách tăng dần
    neighbors = distance[:k]

    values = [neighbor[1] for neighbor in neighbors]
    prediction = sum(values) / len(values)
    return prediction

def knn_mse(x_train, y_train, x_test, y_test, k=4):
    mse = 0.0
    for i in range(len(x_test)):
        pred = knn_regression(x_train, y_train, x_test[i], k)
        mse += (pred - y_test[i]) ** 2
    mse /= len(x_test)
    return mse


np.random.seed(0)
N = 10000
X = np.random.randint(0, 101, size=(N, 2)).astype(float)


noise = np.random.normal(0, 10, size=N)
y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 5.0 + noise #hàm tuyến tính + noise

idx = np.random.permutation(N)
split = int(0.8 * N)
x_train, y_train = X[idx[:split]], y[idx[:split]]
x_valid, y_valid = X[idx[split:]], y[idx[split:]]

k_list = [1, 3, 5, 11, 21, 41, 61, 81, 101, 121]

best_k = None
best_mse = float("inf")

train_mse_list = []
valid_mse_list = []

for k in k_list:
    mse_train = knn_mse(x_train, y_train, x_train, y_train, k)
    mse_valid = knn_mse(x_train, y_train, x_valid, y_valid, k)

    train_mse_list.append(mse_train)
    valid_mse_list.append(mse_valid)

    
    if mse_valid < best_mse:
        best_mse = mse_valid
        best_k = k

print(f"\nBest k = {best_k}, valid MSE = {best_mse:.4f}")

plt.plot(k_list, train_mse_list, marker='o', label='Train MSE')
plt.plot(k_list, valid_mse_list, marker='s', label='Valid MSE')
plt.xlabel('k')
plt.ylabel('MSE')
plt.legend()
plt.show()
