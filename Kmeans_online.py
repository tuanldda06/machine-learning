from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(14)

def kmeans_init_centroids_online(X, K):
    X = np.asarray(X)
    n = X.shape[0]
    first = np.random.randint(0, n)
    centers = [X[first]]
    D = cdist(X, np.asarray(centers))[:, 0]
    for _ in range(1, K):
        probs = D ** 2
        s = probs.sum()
        if s <= 1e-12:
            remain = np.random.choice(n, K - len(centers), replace=False)
            centers.extend(X[remain])
            break
        probs /= s
        idx = np.random.choice(n, p=probs)
        centers.append(X[idx])
        d_new = cdist(X, X[idx][None, :])[:, 0]
        D = np.minimum(D, d_new)
    return np.asarray(centers)

def kmeans_assign_centroids(X, C):
    D = cdist(X, C)
    return np.argmin(D, axis=1)

def kmeans_update_centers(X, label, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[label == k]
        if Xk.shape[0] > 0:
            centers[k] = Xk.mean(axis=0)
    return centers

def has_converged(centers, new_centers, tol=1e-6):
    return np.allclose(centers, new_centers, rtol=0.0, atol=tol)

def main_kmeans(X, K):
    centers_list = [kmeans_init_centroids_online(X, K)]
    labels_list = []
    it = 0
    while True:
        labels = kmeans_assign_centroids(X, centers_list[-1])
        labels_list.append(labels)
        new_centers = kmeans_update_centers(X, labels, K)
        if has_converged(centers_list[-1], new_centers, tol=1e-6):
            break
        centers_list.append(new_centers)
        it += 1
    return centers_list, labels_list, it

X_train = np.r_[
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
    np.random.randn(100, 2) + [0, 5]
]
K = 3
centers, labels, it = main_kmeans(X_train, K)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=labels[-1], s=15)
plt.scatter(centers[-1][:, 0], centers[-1][:, 1], marker='x', s=200, c='red', label='centers')
plt.legend()
plt.title(f'KMeans train, {it} iters')

np.random.seed(42)
X_test = np.r_[
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5],
    np.random.randn(50, 2) + [0, 5]
]
centers_test, labels_test, it_test = main_kmeans(X_test, K)

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=labels_test[-1], s=15)
plt.scatter(centers_test[-1][:, 0], centers_test[-1][:, 1], marker='x', s=200, c='red', label='centers')
plt.legend()
plt.title(f'KMeans online test, {it_test} iters')

plt.tight_layout()
plt.show()
