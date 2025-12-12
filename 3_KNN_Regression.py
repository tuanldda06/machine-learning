import numpy as np

def euclid_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

def knn_regression(x_train , y_train , x_test , k = 4):
    distance = []
    for i in range(len(x_train)) :
        dist = euclid_distance(x_train[i],x_test)
        distance.append((dist , y_train[i]))
    distance.sort(key=lambda x : x[0])
    neighbors = distance[:k]
    neighbor = [neighbor[1] for neighbor in neighbors]
    prediction= sum(neighbor) / len(neighbor)
    return prediction
