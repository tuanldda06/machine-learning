import numpy as np
from collections import Counter

def euclid_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
def knn_predict(x_train , y_train , x_test , k = 3):
    distance = []
    for i in range(len(x_train)):
        dist = euclid_distance(x_train[i] , x_test)
        distance.append((dist , y_train[i]))
    distance.sort(key=lambda x: x[0])# sắp xếp theo khoảng cách tăng dần
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
x_train = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]])
y_train = np.array([0, 0, 0, 1, 1, 2])
x_test = np.array([[1, 1], [8, 9]])
y_test = np.array([0, 2])
k = 3
accuracy_score = accuracy(x_train, y_train, x_test, y_test, k)
print(f"Accuracy: {accuracy_score * 100}%")
for i in range(len(x_test)):
    print(f"Predicted class for {x_test[i]}: {knn_predict(x_train, y_train, x_test[i], k)}")