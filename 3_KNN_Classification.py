import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def euclid_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def knn_predict(x_train , y_train , x_test , k = 3): 
    distance = []
    for i in range(len(x_train)):
        dist = euclid_distance(x_train[i] , x_test)
        distance.append((dist , y_train[i]))
    distance.sort(key=lambda x: x[0])        # sắp xếp theo khoảng cách tăng dần
    neighbors = distance[:k]

    output = [neighbor[1] for neighbor in neighbors]
    prediction = Counter(output).most_common(1)[0][0]
    return prediction

def accuracy(x_train, y_train, x_test, y_test, k=3):
    correct = 0
    for i in range(len(x_test)):
        prediction = knn_predict(x_train, y_train, x_test[i], k)
        if prediction == y_test[i]:
            correct += 1
    return correct / len(x_test)

# ------------------ TẠO DATA GIẢ ------------------
np.random.seed(0)
N= 10000
X = np.random.randint(0, 101, size=(N, 2))

def make_label(x):
    s = x[0] + x[1]
    if s < 70:
         return 0
    elif s < 130: 
        return 1 
    else: 
        return 2 

y = np.array([make_label(p) for p in X])

idx = np.random.permutation(N)
split = int(0.8 * N)
x_train, y_train = X[idx[:split]], y[idx[:split]]
x_valid, y_valid = X[idx[split:]], y[idx[split:]]

# ------------------ CHỌN k ------------------
k_list = [1, 3, 5, 11, 21, 41, 61, 81, 101, 121]

best_k = None
best_acc = -1.0

train_acc_list = []
valid_acc_list = []

for k in k_list:
    acc_train = accuracy(x_train, y_train, x_train, y_train, k)
    acc_valid = accuracy(x_train, y_train, x_valid, y_valid, k)
    train_acc_list.append(acc_train)
    valid_acc_list.append(acc_valid)

    print(f"k = {k:3d} | train_acc = {acc_train:.4f} | valid_acc = {acc_valid:.4f}")

    if acc_valid > best_acc:
        best_acc = acc_valid
        best_k = k

print(f"\nBest k = {best_k}, valid accuracy = {best_acc:.4f}")

# ------------------ VẼ ACCURACY THEO k ------------------
plt.plot(k_list, train_acc_list, marker='o', label='Train acc')
plt.plot(k_list, valid_acc_list, marker='s', label='Valid acc')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
